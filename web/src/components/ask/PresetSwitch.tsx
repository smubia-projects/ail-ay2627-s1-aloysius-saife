import type { GuardConfig, PresetName } from '../../api/types'
import { DETECTOR } from '../../lib/status'

// The strict preset needs Anthropic web search, which this showcase does not have.
const NAMES: PresetName[] = ['fast', 'balanced']
const BLURB: Record<PresetName, string> = {
  fast: 'Quick check, no repair',
  balanced: 'Default: checks and repairs',
  strict: 'High stakes: blocks on any contradiction',
}

interface Props {
  value: PresetName
  presets: Record<string, GuardConfig>
  onChange: (p: PresetName) => void
}

export function PresetSwitch({ value, presets, onChange }: Props) {
  const cfg = presets[value]
  return (
    <fieldset>
      <legend className="text-sm font-medium text-ink-900">Preset</legend>
      <div className="mt-1.5 grid grid-cols-2 gap-1 rounded-md border border-ink-200 bg-ink-50 p-1">
        {NAMES.map((n) => (
          <label
            key={n}
            className={`cursor-pointer rounded px-2 py-1.5 text-center text-sm font-medium capitalize transition-colors duration-150 focus-within:outline focus-within:outline-2 focus-within:outline-action ${
              value === n ? 'bg-white text-ink-900 shadow-panel ring-1 ring-ink-300' : 'text-ink-600 hover:text-ink-900'
            }`}
          >
            <input type="radio" name="preset" value={n} checked={value === n} onChange={() => onChange(n)} className="sr-only" />
            {n}
          </label>
        ))}
      </div>
      <p className="mt-1.5 text-xs text-ink-600">
        {BLURB[value]}
        {cfg
          ? ` · ${cfg.detectors.map((d) => DETECTOR[d]?.label ?? d).join(', ')} · up to ${cfg.budgets.max_passes} pass${cfg.budgets.max_passes === 1 ? '' : 'es'}`
          : ''}
      </p>
    </fieldset>
  )
}
