import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { ApiError, streamRun } from '../api/client'
import type { RunRequest, Source } from '../api/types'
import { loadDemoRun, replayDemo } from '../demo/replay'
import { initialRunState, runReducer } from './runState'

/** 429: the showcase limit was reached (CTA). 503: the demo is paused (notice). */
export type Quota = { kind: 'limit'; count: number } | { kind: 'paused'; message: string } | null

export function useRun(demo: boolean) {
  const [state, dispatch] = useReducer(runReducer, initialRunState)
  const [quota, setQuota] = useState<Quota>(null)
  const stop = useRef<(() => void) | null>(null)

  const cancel = useCallback(() => {
    stop.current?.()
    stop.current = null
  }, [])

  useEffect(() => cancel, [cancel])

  const start = useCallback(
    async (req: RunRequest) => {
      cancel()
      if (demo) {
        const run = await loadDemoRun()
        dispatch({ type: 'connect', sources: run.sources, question: run.question, preset: run.preset })
        stop.current = replayDemo(run, (event) => dispatch({ type: 'event', event }))
        return
      }

      const sources: Source[] = (req.sources ?? []).map((s) => ({ ...s, title: s.title ?? null, url: null }))
      dispatch({ type: 'connect', sources, question: req.question ?? null, preset: req.preset })
      stop.current = streamRun(req, {
        onEvent: (event) => dispatch({ type: 'event', event }),
        onError: (err) => {
          // The AI step stops here: no recorded or substitute run is shown after a 429/503.
          if (err instanceof ApiError && err.status === 429) {
            setQuota({ kind: 'limit', count: err.queriesUsed ?? 0 })
            dispatch({ type: 'error', message: 'This demo has reached its check limit for now.' })
          } else if (err instanceof ApiError && err.status === 503) {
            setQuota({ kind: 'paused', message: err.message })
            dispatch({ type: 'error', message: 'This demo is paused for now.' })
          } else if (err instanceof ApiError) {
            dispatch({ type: 'error', message: `The server rejected the run: ${err.message}` })
          } else if (err instanceof TypeError) {
            dispatch({ type: 'error', message: 'Could not reach the sAIfe server. Try again in a moment.' })
          } else {
            dispatch({ type: 'error', message: err.message })
          }
        },
      })
    },
    [cancel, demo],
  )

  const reset = useCallback(() => {
    cancel()
    dispatch({ type: 'reset' })
  }, [cancel])

  const dismissQuota = useCallback(() => setQuota(null), [])

  return { state, start, reset, quota, dismissQuota }
}
