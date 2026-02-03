window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}



// ===== Trace Viewer (JSONL) + Collapsible =====



// ===== Trace Viewer (JSONL) + Collapsible + Per-example start step =====
// ===== Trace Viewer (JSONL) + Collapsible + Hide prefix text (per example) =====
document.addEventListener("DOMContentLoaded", () => {
  const byId = (id) => document.getElementById(id);

  // ---------- Collapsible shell ----------
  const wrap = byId("traceCollapsible");
  const header = byId("traceCollapsibleHeader");
  const hint = byId("traceCollapsibleHint");

  if (wrap && header && hint) {
    const setState = (expanded) => {
      wrap.classList.toggle("is-expanded", expanded);
      wrap.classList.toggle("is-collapsed", !expanded);
      wrap.setAttribute("aria-expanded", expanded ? "true" : "false");
      hint.textContent = expanded ? "Click to collapse" : "Click to expand";
    };

    setState(false);

    const toggle = (e) => {
      e.preventDefault();
      e.stopPropagation();
      const expanded = wrap.classList.contains("is-expanded");
      setState(!expanded);
    };
    
    header.addEventListener("click", toggle);
    hint.addEventListener("click", toggle);

    wrap.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        const expanded = wrap.classList.contains("is-expanded");
        setState(!expanded);
      }
      if (e.key === "Escape") setState(false);
    });

    document.addEventListener("pointerdown", (e) => {
      if (!wrap.contains(e.target)) setState(false);
    });
  }

  // ---------- Trace Viewer ----------
  const statusEl = byId("traceStatus");
  const stepSlider = byId("traceStepSlider");
  const stepLabel = byId("traceStepLabel");
  const playBtn = byId("tracePlayBtn");
  const metaRow = byId("traceMetaRow");

  const beforeBox = byId("traceBeforeBox");
  const afterBox = byId("traceAfterBox");
  const beforeInfo = byId("traceBeforeInfo");
  const afterInfo = byId("traceAfterInfo");

  const m2uCount = byId("traceM2UCount");
  const revCount = byId("traceRevCount");

  const blockLabel = byId("traceBlockLabel");
  const innerStepLabel = byId("traceInnerStepLabel");
  const sampleLabel = byId("traceSampleLabel");

  const exampleBtns = document.querySelectorAll("#trace-viewer .trace-example-btn");

  if (!statusEl || !stepSlider || !stepLabel || !playBtn || !beforeBox || !afterBox) return;

  // >>> EDIT THESE VALUES <<<
  // Character cutoffs: hide first N chars from the displayed before/after text per example.
  const TEXT_CUTOFF_BY_EXAMPLE = {
    1: { before: 1504, after: 1504 },
    2: { before: 1504, after: 1504 },
    3: { before: 2213, after: 2213 },
    4: { before: 1801, after: 1801 },
  };

  // state
  let records = [];
  let cur = 0;
  let playing = false;
  let playTimer = null;
  let isScrubbing = false;
  let currentExample = 1;

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

  function countMaskTokens(str) {
    const m = String(str).match(/\[MASK\]/g);
    return m ? m.length : 0;
  }

  /**
   * Trim text prefix and shift spans so highlights still align.
   * @param {string} text
   * @param {Array<{start:number,end:number,cls:string}>} spans
   * @param {number} cut - number of chars to cut from front
   * @returns {{text:string, spans:Array}}
   */
  function trimTextAndShiftSpans(text, spans, cut) {
    text = String(text ?? "");
    cut = Math.max(0, Math.min(cut || 0, text.length));

    const trimmedText = text.slice(cut);

    const shifted = (spans || []).map(sp => {
      const s = (sp.start ?? 0) - cut;
      const e = (sp.end ?? 0) - cut;
      return { start: s, end: e, cls: sp.cls };
    });

    // We'll clamp in renderWithSpans anyway; filter fully-negative spans for cleanliness.
    const filtered = shifted.filter(sp => sp.end > 0);

    return { text: trimmedText, spans: filtered, cut };
  }

  /**
   * Render with character-level span priority (earlier spans win).
   */
  function renderWithSpans(text, spans, options) {
    text = String(text ?? "");
    const n = text.length;
    const clsArr = new Array(n).fill(null);

    for (const sp of spans) {
      const s = clamp(sp.start, 0, n);
      const e = clamp(sp.end, 0, n);
      for (let i = s; i < e; i++) {
        if (clsArr[i] === null) clsArr[i] = sp.cls;
      }
    }

    const dimMask = options?.dimMask ?? true;
    const maskToken = "[MASK]";

    let html = "";
    let i = 0;
    while (i < n) {
      const cls = clsArr[i];

      if (dimMask && cls === null && text.startsWith(maskToken, i)) {
        html += `<span class="hl mask">${escapeHtml(maskToken)}</span>`;
        i += maskToken.length;
        continue;
      }

      let j = i + 1;
      while (j < n && clsArr[j] === cls) j++;

      const chunk = escapeHtml(text.slice(i, j));
      html += cls ? `<span class="hl ${cls}">${chunk}</span>` : chunk;
      i = j;
    }
    return html;
  }

  function setEnabled(on) {
    stepSlider.disabled = !on;
    playBtn.disabled = !on;
    if (metaRow) metaRow.style.display = on ? "flex" : "none";
  }

  async function fetchJsonl(url) {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(`Failed to load ${url} (${res.status})`);
    const text = await res.text();
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);

    const recs = [];
    for (let i = 0; i < lines.length; i++) {
      try {
        recs.push(JSON.parse(lines[i]));
      } catch (e) {
        console.warn("Bad JSON on line", i + 1, url, e);
      }
    }

    recs.sort((a, b) => ((a.block ?? 0) - (b.block ?? 0)) || ((a.inner_step ?? 0) - (b.inner_step ?? 0)));
    return recs;
  }

  function stopPlaying() {
    playing = false;
    playBtn.textContent = "▶ Play";
    if (playTimer) clearInterval(playTimer);
    playTimer = null;
  }

  function startPlaying() {
    if (!records.length) return;
    playing = true;
    playBtn.textContent = "⏸ Pause";

    const fps = 12;
    playTimer = setInterval(() => {
      if (isScrubbing) return;
      if (cur >= records.length - 1) {
        stopPlaying();
        return;
      }
      cur++;
      stepSlider.value = String(cur);
      render();
    }, 1000 / fps);
  }

  function render() {
    if (!records.length) return;
    const rec = records[cur];

    stepLabel.textContent = `${cur + 1} / ${records.length}`;
    if (blockLabel) blockLabel.textContent = rec.block ?? "—";
    if (innerStepLabel) innerStepLabel.textContent = rec.inner_step ?? "—";
    if (sampleLabel) sampleLabel.textContent = rec.sample_idx ?? "—";

    const rawBefore = rec.before ?? "";
    const rawAfter = rec.after ?? "";

    // pick cutoff for current example
    const cutCfg = TEXT_CUTOFF_BY_EXAMPLE[currentExample] || { before: 0, after: 0 };
    const cutBefore = cutCfg.before || 0;
    const cutAfter = cutCfg.after || 0;

    // prompt ranges (original coordinates)
    const prBefore = rec.prompt_char_range_before || [0, 0];
    const prAfter = rec.prompt_char_range_after || [0, 0];

    // build highlight spans (original coords)
    const spansBefore = [];
    const spansAfter = [];

    // 1) mask->unmask (cyan)
    const m2u = rec.changes?.mask_to_unmask ?? [];
    for (const c of m2u) {
      const [bs, be] = c.before_span || [0, 0];
      const [as, ae] = c.after_span || [0, 0];
      spansBefore.push({ start: bs, end: be, cls: "m2u" });
      spansAfter.push({ start: as, end: ae, cls: "m2u" });
    }

    // 2) revisions (pink)
    const u2m = rec.changes?.unmask_to_mask ?? [];
    const other = rec.changes?.other_token_change ?? [];
    const revisions = u2m.concat(other);
    for (const c of revisions) {
      const [bs, be] = c.before_span || [0, 0];
      const [as, ae] = c.after_span || [0, 0];
      spansBefore.push({ start: bs, end: be, cls: "u2m" });
      spansAfter.push({ start: as, end: ae, cls: "u2m" });
    }

    // 3) prompt (lowest priority)
    spansBefore.push({ start: prBefore[0] || 0, end: prBefore[1] || 0, cls: "prompt" });
    spansAfter.push({ start: prAfter[0] || 0, end: prAfter[1] || 0, cls: "prompt" });

    // Trim text + shift spans
    const tb = trimTextAndShiftSpans(rawBefore, spansBefore, cutBefore);
    const ta = trimTextAndShiftSpans(rawAfter, spansAfter, cutAfter);

    beforeBox.innerHTML = renderWithSpans(tb.text, tb.spans, { dimMask: true });
    afterBox.innerHTML = renderWithSpans(ta.text, ta.spans, { dimMask: true });

    if (m2uCount) m2uCount.textContent = String(m2u.length);
    if (revCount) revCount.textContent = String(revisions.length);

    // Info line should reflect displayed text, but you can also show hidden amount.
    const bMasks = countMaskTokens(tb.text);
    const aMasks = countMaskTokens(ta.text);

    if (beforeInfo) beforeInfo.textContent =
      `${tb.text.length.toLocaleString()} chars • ${bMasks.toLocaleString()} [MASK] • hidden ${tb.cut.toLocaleString()}`;
    if (afterInfo) afterInfo.textContent =
      `${ta.text.length.toLocaleString()} chars • ${aMasks.toLocaleString()} [MASK] • hidden ${ta.cut.toLocaleString()}`;
  }

  async function loadExample(exampleNum) {
    currentExample = Number(exampleNum);
    stopPlaying();
    setEnabled(false);
    statusEl.textContent = `Loading Example ${exampleNum}…`;

    const url = `static/examples/example_${exampleNum}.jsonl`;
    try {
      records = await fetchJsonl(url);

      if (!records.length) {
        statusEl.textContent = `Example ${exampleNum} loaded, but no valid JSONL lines found.`;
        setEnabled(false);
        return;
      }

      cur = 0;

      stepSlider.min = "0";
      stepSlider.max = String(records.length - 1);
      stepSlider.value = "0";

      statusEl.textContent = `Example ${exampleNum} • ${records.length} steps`;
      setEnabled(true);
      render();
      startPlaying();
    } catch (e) {
      console.error(e);
      statusEl.textContent = `Failed to load Example ${exampleNum}. Check that ${url} exists.`;
      setEnabled(false);
    }
  }

  // buttons
  exampleBtns.forEach(btn => {
    btn.addEventListener("click", async () => {
      const n = Number(btn.getAttribute("data-example"));
      exampleBtns.forEach(b => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      await loadExample(n);
    });
  });

  // play/pause
  playBtn.addEventListener("click", () => {
    if (!records.length) return;
    if (playing) stopPlaying();
    else startPlaying();
  });

  // slider scrubbing pauses + follows
  const beginScrub = () => { isScrubbing = true; stopPlaying(); };
  const endScrub = () => { isScrubbing = false; };

  stepSlider.addEventListener("pointerdown", beginScrub);
  stepSlider.addEventListener("mousedown", beginScrub);
  stepSlider.addEventListener("touchstart", beginScrub, { passive: true });

  stepSlider.addEventListener("pointerup", endScrub);
  stepSlider.addEventListener("mouseup", endScrub);
  stepSlider.addEventListener("touchend", endScrub);

  stepSlider.addEventListener("input", () => {
    if (!records.length) return;
    cur = parseInt(stepSlider.value, 10) || 0;
    render();
  });

  // default
  loadExample(1);
});
