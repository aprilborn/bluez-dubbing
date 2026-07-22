import { state } from "./state.js";
import { el } from "./dom.js";

// Speaker picker shown only for TTS models that expose a speaker catalog
// (currently silero). The available voices track the primary target language.
export const ttsSpeaker = {
  update() {
    const field = el.ttsSpeakerField;
    const select = el.ttsSpeaker;
    const modelSelect = el.ttsModel;
    if (!field || !select || !modelSelect) return;

    const catalog = state.ttsSpeakers?.[modelSelect.value] || null;
    const primaryLang = (state.targetLangs[0] || "").toLowerCase();
    const speakers = catalog ? catalog[primaryLang] || [] : [];

    if (!speakers.length) {
      field.hidden = true;
      select.value = "";
      return;
    }

    const prev = select.value;
    const frag = document.createDocumentFragment();
    frag.appendChild(new Option("Auto (per speaker)", ""));
    speakers.forEach(name => frag.appendChild(new Option(name, name)));
    select.replaceChildren(frag);
    select.value = speakers.includes(prev) ? prev : "";
    field.hidden = false;
  }
};
