import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionProfileRoute } from '@/store/session-request-router'

const directTranscribe = vi.hoisted(() => vi.fn())
const relayTranscribe = vi.hoisted(() => vi.fn())
const blobToDataUrl = vi.hoisted(() => vi.fn(async () => 'data:audio/webm;base64,dGlsZQ=='))

vi.mock('@/lib/voice-client-direct', () => ({
  transcribeAudioClientDirect: directTranscribe
}))

vi.mock('@/hermes', () => ({
  transcribeAudio: relayTranscribe
}))

vi.mock('@/app/session/hooks/use-prompt-actions/utils', () => ({
  blobToDataUrl
}))

const { transcribeSessionTileAudio } = await import('./session-tile-transcription')

const ownerRoute: SessionProfileRoute = {
  connectionId: 'registry-owner',
  mode: 'remote',
  profile: 'desktop-alias',
  targetProfile: 'backend-worker'
}

describe('session tile transcription routing', () => {
  afterEach(() => vi.clearAllMocks())

  it("fetches direct STT configuration from the tile owner's backend scope", async () => {
    directTranscribe.mockResolvedValue('owner heard this')
    const audio = new Blob(['voice'], { type: 'audio/webm' })

    await expect(transcribeSessionTileAudio(audio, ownerRoute)).resolves.toBe('owner heard this')

    expect(directTranscribe).toHaveBeenCalledWith(audio, {
      connectionId: 'registry-owner',
      profile: 'backend-worker'
    })
    expect(relayTranscribe).not.toHaveBeenCalled()
  })

  it('relays through the same owner scope when direct STT is unavailable', async () => {
    directTranscribe.mockResolvedValue(null)
    relayTranscribe.mockResolvedValue({ transcript: 'owner relay heard this' })
    const audio = new Blob(['voice'], { type: 'audio/webm' })

    await expect(transcribeSessionTileAudio(audio, ownerRoute)).resolves.toBe('owner relay heard this')

    expect(relayTranscribe).toHaveBeenCalledWith('data:audio/webm;base64,dGlsZQ==', 'audio/webm', {
      connectionId: 'registry-owner',
      profile: 'backend-worker'
    })
  })
})
