import { useEffect, useState } from 'react'
import { api, probeApi } from './api/client'
import type { GuardConfig, PresetName, SourceInput, Trap } from './api/types'
import { CheckPage } from './CheckPage'
import { Header } from './components/Header'
import { ResultsPage } from './components/results/ResultsPage'
import { DEMO_PRESETS, DEMO_TRAPS } from './demo/fixtures'
import { loadDemoRun } from './demo/replay'
import { demoRequested, useRoute } from './lib/route'

type Mode =
  | { kind: 'probing' }
  | { kind: 'live'; presets: Record<string, GuardConfig> }
  | { kind: 'demo'; reason: 'requested' | 'offline' }

type Initial = { question: string; preset: PresetName; sources: SourceInput[] }

export function App() {
  const [route, navigate] = useRoute()
  const [mode, setMode] = useState<Mode>(() => (demoRequested() ? { kind: 'demo', reason: 'requested' } : { kind: 'probing' }))
  const [traps, setTraps] = useState<Trap[] | null>(null)
  const [trapsError, setTrapsError] = useState<string | null>(null)
  const [initial, setInitial] = useState<Initial | undefined>(undefined)

  useEffect(() => {
    if (mode.kind !== 'probing') return
    let alive = true
    probeApi().then((presets) => {
      if (alive) setMode(presets ? { kind: 'live', presets } : { kind: 'demo', reason: 'offline' })
    })
    return () => {
      alive = false
    }
  }, [mode.kind])

  useEffect(() => {
    if (mode.kind === 'live') {
      api.traps().then(setTraps, () => setTrapsError('Could not load the trap questions from the server.'))
    } else if (mode.kind === 'demo') {
      setTraps(DEMO_TRAPS)
      loadDemoRun().then((run) =>
        setInitial({
          question: run.question,
          preset: (['fast', 'balanced'].includes(run.preset) ? run.preset : 'balanced') as PresetName,
          sources: run.sources.map((s) => ({ id: s.id, text: s.text, title: s.title ?? undefined })),
        }),
      )
    }
  }, [mode.kind])

  const demo = mode.kind === 'demo'
  const ready = mode.kind === 'live' || (demo && initial)

  return (
    <div className="min-h-screen">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <Header route={route} onNavigate={navigate} demo={demo} demoReason={demo ? mode.reason : null} />
      <div id="main-content">
        {route === 'results' ? (
          <ResultsPage mode={mode.kind} />
        ) : ready ? (
          <CheckPage
            key={mode.kind}
            demo={demo}
            presets={mode.kind === 'live' ? mode.presets : DEMO_PRESETS}
            traps={traps}
            trapsError={trapsError}
            initial={initial}
          />
        ) : (
          <p className="px-6 py-10 text-sm text-ink-600" aria-live="polite">
            Connecting to the sAIfe server…
          </p>
        )}
      </div>
    </div>
  )
}
