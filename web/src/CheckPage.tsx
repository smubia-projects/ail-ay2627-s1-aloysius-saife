import { useState } from 'react'
import type { GuardConfig, PresetName, RunRequest, SourceInput, Trap } from './api/types'
import { AlertBanner } from './components/AlertBanner'
import { AnswerView } from './components/AnswerView'
import { AskPanel } from './components/ask/AskPanel'
import { DiffView } from './components/DiffView'
import { EmptyState } from './components/EmptyState'
import { EvidenceDrawer } from './components/EvidenceDrawer'
import { PassTimeline } from './components/PassTimeline'
import { DemoPausedNotice, RateLimitCTA } from './components/RateLimitCTA'
import { RunFooter } from './components/RunFooter'
import { useRun } from './lib/useRun'

interface Props {
  demo: boolean
  presets: Record<string, GuardConfig>
  traps: Trap[] | null
  trapsError: string | null
  initial?: { question: string; preset: PresetName; sources: SourceInput[] }
}

type Selection = { claimId: string; passIndex: number } | null

export function CheckPage({ demo, presets, traps, trapsError, initial }: Props) {
  const { state, start, reset, quota, dismissQuota } = useRun(demo)
  const [selection, setSelection] = useState<Selection>(null)
  const busy = state.phase === 'connecting' || state.phase === 'running'
  const lastPass = Math.max(0, state.passes.length - 1)

  const submit = (req: RunRequest) => {
    setSelection(null)
    void start(req)
  }
  const clear = () => {
    setSelection(null)
    reset()
  }
  const selectLatest = (claimId: string) => setSelection({ claimId, passIndex: lastPass })

  return (
    <div className="mx-auto grid max-w-[90rem] gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(20rem,26rem)_minmax(0,1fr)] lg:items-start">
      <div className="lg:sticky lg:top-5">
        <AskPanel
          presets={presets}
          traps={traps}
          trapsError={trapsError}
          busy={busy}
          demo={demo}
          initial={initial}
          onSubmit={submit}
          onReset={clear}
          canReset={state.phase !== 'idle'}
        />
      </div>

      <main className="min-w-0 space-y-5" aria-label="Run">
        {state.phase === 'idle' ? (
          <EmptyState demo={demo} />
        ) : (
          <>
            <AlertBanner state={state} onSelectClaim={selectLatest} />
            <AnswerView
              state={state}
              selectedId={selection?.claimId ?? null}
              onSelect={(claimId, passIndex) => setSelection({ claimId, passIndex })}
            />
            <PassTimeline state={state} selected={selection} onSelect={(claimId, passIndex) => setSelection({ claimId, passIndex })} />
            {state.report ? <DiffView report={state.report} onSelect={selectLatest} /> : null}
            {state.report ? <RunFooter report={state.report} /> : null}
          </>
        )}
      </main>

      <EvidenceDrawer state={state} selection={selection} onClose={() => setSelection(null)} />
      {quota?.kind === 'limit' ? <RateLimitCTA queriesMade={quota.count} onDismiss={dismissQuota} /> : null}
      {quota?.kind === 'paused' ? <DemoPausedNotice message={quota.message} onDismiss={dismissQuota} /> : null}
    </div>
  )
}
