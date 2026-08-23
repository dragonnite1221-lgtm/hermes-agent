import type { ProfileScope } from '@/api/client'
import { blobToDataUrl } from '@/app/session/hooks/use-prompt-actions/utils'
import { transcribeAudio } from '@/hermes'
import { transcribeAudioClientDirect } from '@/lib/voice-client-direct'
import type { SessionProfileRoute } from '@/store/session-request-router'

function tileVoiceScope(ownerRoute?: SessionProfileRoute): ProfileScope {
  return ownerRoute
    ? {
        connectionId: ownerRoute.connectionId,
        profile: ownerRoute.targetProfile || ownerRoute.profile
      }
    : undefined
}

/** Transcribe against the backend that owns the tile, even when another
 *  profile is active in the foreground. Both direct config lookup and the
 *  relay fallback share the same immutable owner scope. */
export async function transcribeSessionTileAudio(audio: Blob, ownerRoute?: SessionProfileRoute): Promise<string> {
  const scope = tileVoiceScope(ownerRoute)
  const direct = await transcribeAudioClientDirect(audio, scope)

  if (direct !== null) {
    return direct
  }

  return (await transcribeAudio(await blobToDataUrl(audio), audio.type, scope)).transcript
}
