// LLM Polish prompt editor.
//
// Lets the user view/edit the instructions the `llm_polish` translation model
// sends to the local Ollama model, persisted in localStorage. The current values
// are shipped to the backend on submit (see handlers.submitForm) and reach the
// runner via TranslateRequest.extra. {{placeholders}} are substituted server-side
// and MUST be preserved by the user.
//
// KEEP IN SYNC: backend/services/translation/models/llm_polish/runner.py holds
// the authoritative defaults; a blank textarea falls back to the runner default.
const DEFAULTS = {
  system_prompt:
    "You are a professional subtitle localization editor. You receive a full " +
    "list of subtitle segments for one video, each with the original text and a " +
    "rough machine-translated draft. You improve the draft translations as a " +
    "single conversation: fix cross-segment pronoun and reference consistency, " +
    "remove overly literal or awkward phrasing, and make every line natural and " +
    "idiomatic in the target language while preserving the exact meaning. You " +
    "MUST NOT merge, split, reorder, add or drop segments. Return one improved " +
    "line per input segment, keyed by its id. Respond with JSON only.",

  context_template:
    "For continuity, here are the already-finalized translations of the " +
    "immediately preceding segments. Do NOT re-output them; use them only " +
    "to keep pronouns, references and terminology consistent:\n" +
    "{{context_pairs}}\n\n",

  user_template:
    "Target language (ISO code): {{target_lang}}\n" +
    "Source language (ISO code): {{source_lang}}\n\n" +
    "{{context_block}}" +
    "Improve the following subtitle segments. Each has an integer `id`, the " +
    "`original` source text and a machine-translated `draft`:\n\n" +
    "{{segments}}\n\n" +
    "Return exactly one improved entry per id shown above, reusing the same ids.\n" +
    "Rules:\n" +
    "- Treat `original` (the source-language text) as the source of truth for " +
    "meaning. If the `draft` mistranslates, omits or distorts `original`, correct " +
    "the translation so it faithfully conveys `original`.\n" +
    "- Only output the translated text; never copy `original` verbatim.\n" +
    "- Fix pronouns/references so they stay consistent across segments.\n" +
    "- Prefer natural, idiomatic phrasing over literal word-for-word.\n" +
    "- Never merge, split, add or drop segments.",
};

const FIELDS = {
  system_prompt: "llm-system-prompt",
  context_template: "llm-context-template",
  user_template: "llm-user-template",
};

export const llmPrompts = {
  KEY: "bluez-llm-polish-prompts",
  DEFAULTS,

  load() {
    let saved = {};
    try {
      saved = JSON.parse(localStorage.getItem(this.KEY) || "{}") || {};
    } catch {
      saved = {};
    }
    return { ...DEFAULTS, ...saved };
  },

  save(values) {
    try {
      localStorage.setItem(this.KEY, JSON.stringify(values));
    } catch {
      /* localStorage unavailable (private mode); edits stay in the DOM only */
    }
  },

  fields() {
    return Object.fromEntries(
      Object.entries(FIELDS).map(([key, id]) => [key, document.getElementById(id)])
    );
  },

  // Current textarea values, falling back to the default when a box is blank.
  get() {
    const els = this.fields();
    const out = {};
    for (const key of Object.keys(FIELDS)) {
      const val = (els[key]?.value ?? "").trim();
      out[key] = val || DEFAULTS[key];
    }
    return out;
  },

  populate(values) {
    const els = this.fields();
    for (const key of Object.keys(FIELDS)) {
      if (els[key]) els[key].value = values[key] ?? DEFAULTS[key];
    }
  },

  init() {
    const els = this.fields();
    if (!Object.values(els).some(Boolean)) return; // section not present

    this.populate(this.load());

    const persist = () => {
      const current = {};
      for (const key of Object.keys(FIELDS)) current[key] = els[key]?.value ?? "";
      this.save(current);
    };

    for (const key of Object.keys(FIELDS)) {
      els[key]?.addEventListener("input", persist);
    }

    const resetBtn = document.getElementById("llm-prompts-reset");
    if (resetBtn) {
      resetBtn.onclick = () => {
        this.populate({ ...DEFAULTS });
        this.save({ ...DEFAULTS });
      };
    }
  },
};
