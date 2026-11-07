import type { ResultSummary } from '../components/results/types'
import type { GuardConfig, RunEvent, RunRequest, Trap } from './types'
import { EVENT_TYPES } from './types'

// The Cloud Run API in production (VITE_API_BASE, set on Vercel); the Vite dev proxy otherwise.
const API_ROOT = ((import.meta.env.VITE_API_BASE as string | undefined) ?? '').replace(/\/+$/, '')
const BASE = `${API_ROOT}/api`

export class ApiError extends Error {
  readonly status: number
  /** From the rate limiter's 429 body: the limit that was reached. */
  readonly queriesUsed: number | null
  constructor(status: number, message: string, queriesUsed: number | null = null) {
    super(message)
    this.status = status
    this.queriesUsed = queriesUsed
  }
}

/** Reads FastAPI's error body: ``detail`` is a string, or ``{message, queries_used}`` from the limiter. */
async function errorFrom(res: Response): Promise<ApiError> {
  let detail = res.statusText
  let used: number | null = null
  try {
    const body = (await res.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') detail = body.detail
    else if (body.detail && typeof body.detail === 'object') {
      const d = body.detail as { message?: unknown; queries_used?: unknown }
      if (typeof d.message === 'string') detail = d.message
      if (typeof d.queries_used === 'number') used = d.queries_used
    }
  } catch {
    // non-JSON error body; keep the status text
  }
  return new ApiError(res.status, detail || `HTTP ${res.status}`, used)
}

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const ctrl = new AbortController()
  const timer = window.setTimeout(() => ctrl.abort(), init?.timeoutMs ?? 10_000)
  try {
    const res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: ctrl.signal,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}) },
    })
    if (!res.ok) throw await errorFrom(res)
    const type = res.headers.get('content-type') ?? ''
    if (!type.includes('json')) throw new ApiError(res.status, 'The server did not return JSON.')
    return (await res.json()) as T
  } finally {
    window.clearTimeout(timer)
  }
}

export const api = {
  // A cold Cloud Run start can take 10 to 30 s; don't mistake it for an offline server.
  presets: () => request<Record<string, GuardConfig>>('/presets', { timeoutMs: 20_000 }),
  traps: () => request<Trap[]>('/traps'),
  /** Benchmark summaries from bench/results/*.json, newest first. */
  results: () => request<ResultSummary[]>('/results'),
}

export interface StreamHandlers {
  onEvent: (event: RunEvent) => void
  onError: (err: ApiError | Error) => void
}

const KNOWN_TYPES = new Set<string>(EVENT_TYPES)

/**
 * Run a check and read its events from the same request (SSE over a POST body).
 * The run lives only for this request, so nothing is fetched afterwards: the
 * run.completed event carries the report. Returns a function that aborts the run.
 */
export function streamRun(body: RunRequest, { onEvent, onError }: StreamHandlers): () => void {
  const ctrl = new AbortController()
  let finished = false

  const handleBlock = (block: string) => {
    const data = block
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).replace(/^ /, ''))
      .join('\n')
    if (!data) return
    let event: RunEvent
    try {
      event = JSON.parse(data) as RunEvent
    } catch {
      return
    }
    if (!KNOWN_TYPES.has(event.type)) return
    onEvent(event)
    if (event.type === 'run.completed' || event.type === 'run.failed') finished = true
  }

  ;(async () => {
    const res = await fetch(`${BASE}/runs/stream`, {
      method: 'POST',
      signal: ctrl.signal,
      headers: { Accept: 'text/event-stream', 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw await errorFrom(res)
    if (!res.body) throw new Error('The server did not stream the run.')
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
    let buffer = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += value.replace(/\r\n?/g, '\n')
      let cut: number
      while ((cut = buffer.indexOf('\n\n')) >= 0) {
        handleBlock(buffer.slice(0, cut))
        buffer = buffer.slice(cut + 2)
      }
    }
    if (buffer.trim()) handleBlock(buffer)
    if (!finished) throw new Error('Lost the connection to the sAIfe server before the run finished.')
  })().catch((err: unknown) => {
    if (ctrl.signal.aborted) return
    onError(err instanceof Error ? err : new Error(String(err)))
  })

  return () => ctrl.abort()
}

/** True when the API answers. Used to fall back to demo mode when it doesn't. */
export async function probeApi(): Promise<Record<string, GuardConfig> | null> {
  try {
    return await api.presets()
  } catch {
    return null
  }
}
