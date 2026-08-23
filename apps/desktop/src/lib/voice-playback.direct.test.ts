import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as VoiceClientDirect from './voice-client-direct'

const directTtsConfig = vi.hoisted(() => vi.fn())
const synthesizeSpeechClientDirect = vi.hoisted(() => vi.fn())
const playAudio = vi.fn<() => Promise<void>>()

vi.mock('@/lib/voice-client-direct', async importOriginal => {
  const actual = await importOriginal<typeof VoiceClientDirect>()

  return {
    ...actual,
    directTtsConfig,
    synthesizeSpeechClientDirect
  }
})

vi.mock('@/hermes', () => ({
  getApiRequestConnection: vi.fn(() => null),
  getApiRequestProfile: vi.fn(() => null),
  speakText: vi.fn()
}))

const { startSpeechStream, stopVoicePlayback } = await import('./voice-playback')

const tts: VoiceClientDirect.DirectTtsConfig = {
  api_key: 'test-key',
  base_url: 'https://tts.example/v1',
  mode: 'direct',
  model: 'tts-model',
  provider: 'test',
  speed: null,
  voice: 'voice',
  wire: 'openai-speech'
}

class FakeAudio extends EventTarget {
  static instances: FakeAudio[] = []

  readonly pause = vi.fn()
  readonly play = playAudio
  src: string

  constructor(src: string) {
    super()
    this.src = src
    FakeAudio.instances.push(this)
  }
}

async function startDirectSession() {
  const session = await startSpeechStream({ messageId: 'message-1', source: 'voice-conversation' })

  expect(session).not.toBeNull()
  session!.append('This sentence is long enough to synthesize immediately.')
  session!.finish()

  await vi.waitFor(() => expect(FakeAudio.instances).toHaveLength(1))

  return session!
}

describe('client-direct voice playback fallback boundary', () => {
  beforeEach(() => {
    FakeAudio.instances = []
    playAudio.mockReset()
    directTtsConfig.mockResolvedValue(tts)
    synthesizeSpeechClientDirect.mockResolvedValue(new Uint8Array([1, 2, 3]).buffer)
    vi.stubGlobal('Audio', FakeAudio)
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:voice'),
      revokeObjectURL: vi.fn()
    })
  })

  afterEach(() => {
    stopVoicePlayback()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
  })

  it('returns fallback when play rejects before audio starts', async () => {
    playAudio.mockRejectedValueOnce(new Error('decode rejected'))

    const session = await startDirectSession()

    await expect(session.done).resolves.toBe('fallback')
  })

  it('returns fallback when corrupt audio emits error before playing', async () => {
    playAudio.mockReturnValueOnce(new Promise(() => undefined))

    const session = await startDirectSession()
    FakeAudio.instances[0].dispatchEvent(new Event('error'))

    await expect(session.done).resolves.toBe('fallback')
  })

  it('does not replay from fallback after audio actually started', async () => {
    playAudio.mockReturnValueOnce(new Promise(() => undefined))

    const session = await startDirectSession()
    FakeAudio.instances[0].dispatchEvent(new Event('playing'))
    FakeAudio.instances[0].dispatchEvent(new Event('error'))

    await expect(session.done).resolves.toBe('done')
  })
})
