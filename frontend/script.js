(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const form = $('#generate-form');
  const promptInput = $('#prompt');
  const generateButton = $('#generate-btn');
  const buildSection = $('#build-section');
  const statusText = $('#status-text');
  const statusPill = $('.build-status');
  const logOutput = $('#log-output');
  const fileTabs = $('#file-tabs');
  const codeOutput = $('#code-output code');
  const activeFilename = $('#active-filename');
  const previewFrame = $('#preview-frame');
  const previewPlaceholder = $('#preview-placeholder');
  const iframeWrap = $('#iframe-wrap');
  const fileByName = new Map();
  let renderedLogCount = 0;
  let currentJobId = null;
  let pollTimer = null;
  let pollInProgress = false;
  let consecutivePollFailures = 0;
  let startedAt = 0;
  let activeFile = null;
  let revealToken = 0;
  let previewUrl = '';
  let shareUrl = '';
  let shareUrlReady = Promise.resolve();
  let lastFileSnapshot = '';
  let buildSeconds = 0;

  const friendlyStep = {
    queued: 'In the queue',
    '3_spec_generation': 'Understanding your idea',
    '5_code_generation': 'Building the app',
    '6_preflight': 'Checking generated files and API routes',
    '6_execution': 'Starting your app',
    '7_verification': 'Checking that it works',
    '8_complete': 'App ready',
    '8_failed': 'Build needs another try',
    error: 'Build needs another try'
  };

  function appendLog(message) {
    const empty = logOutput.querySelector('.log-empty');
    if (empty) empty.remove();
    const row = document.createElement('p');
    row.className = 'log-line';
    const index = document.createElement('span');
    index.className = 'log-index';
    index.textContent = String(logOutput.children.length + 1).padStart(2, '0');
    const text = document.createElement('span');
    text.className = 'log-message';
    text.textContent = message;
    const lower = message.toLowerCase();
    if (lower.includes('passed') || lower.includes('ready')) text.classList.add('is-success');
    if (lower.startsWith('error:') || lower.startsWith('failure:')) text.classList.add('is-error');
    if (lower.includes('self-repair') || lower.includes('switched model')) text.classList.add('is-repair');
    row.append(index, text);
    logOutput.append(row);
    logOutput.scrollTop = logOutput.scrollHeight;
  }

  function clearLogs() {
    logOutput.replaceChildren();
    renderedLogCount = 0;
  }

  function setStatus(label, state = 'busy') {
    statusText.textContent = label;
    statusPill.classList.toggle('is-busy', state === 'busy');
    statusPill.classList.toggle('is-done', state === 'done');
    statusPill.classList.toggle('is-error', state === 'error');
  }

  function clearResult() {
    $('#result-actions').classList.add('hidden');
    $('#stats-strip').classList.add('hidden');
    $('#compare-panel').classList.add('hidden');
    $('#modify-panel').classList.add('hidden');
    $('#repair-summary').classList.add('hidden');
    $('#verification-summary').classList.add('hidden');
    $('#verification-checks').replaceChildren();
    $('#error-panel').classList.add('hidden');
    $('#qr-panel').classList.add('hidden');
    $('#developer-details').open = false;
  }

  function resetWorkspace() {
    clearLogs();
    fileByName.clear();
    fileTabs.replaceChildren();
    const empty = document.createElement('span');
    empty.className = 'file-empty';
    empty.textContent = 'Files appear as ForgeAI creates them';
    fileTabs.append(empty);
    activeFile = null;
    lastFileSnapshot = '';
    revealToken += 1;
    consecutivePollFailures = 0;
    previewUrl = '';
    shareUrl = '';
    shareUrlReady = Promise.resolve();
    codeOutput.className = 'language-plain';
    codeOutput.textContent = '// Generated files will appear here';
    activeFilename.textContent = 'Select a file';
    $('#copy-code').disabled = true;
    $('#file-count').textContent = '0 files';
    $('#preview-state').textContent = 'Waiting for a successful build';
    $('#preview-address').textContent = 'Your app will open here';
    previewFrame.src = 'about:blank';
    previewPlaceholder.classList.remove('hidden');
    $$('#pipeline li').forEach((step) => {
      step.classList.remove('is-done', 'is-active', 'repair-on');
      step.querySelector('.step-icon').textContent = step.dataset.step === 'self-heal' ? '6' : String([...step.parentElement.children].indexOf(step) + 1);
    });
    $('#workspace-grid').classList.remove('hidden');
    $('#blueprint-panel').classList.add('hidden');
    $('#blueprint-tab').setAttribute('aria-expanded', 'false');
    clearResult();
  }

  function renderPipeline(data) {
    const step = data.current_step || 'queued';
    const logs = data.logs || [];
    const verification = data.verification || data.result?.verification || {};
    const repairs = Array.isArray(verification.repairs) ? verification.repairs : [];
    const repairStarted = logs.some((line) => line.toLowerCase().includes('self-repair applied'));
    const current = step === 'queued' || step === '3_spec_generation' ? 0
      : step === '5_code_generation' ? 2
      : step === '6_execution' ? 3
      : step === '7_verification' ? (repairStarted && repairs.length === 0 ? 5 : 4)
      : (step === '8_complete' || data.status === 'success') ? 5
      : step === '8_failed' || data.status === 'failed' ? 4 : 0;
    const completed = data.status === 'success' ? 5 : Math.max(0, current);
    $$('#pipeline li').forEach((item, index) => {
      const isSelfHeal = item.dataset.step === 'self-heal';
      const repaired = isSelfHeal && (repairs.length > 0 || repairStarted);
      item.classList.toggle('is-done', index < completed || (isSelfHeal && repairs.length > 0));
      item.classList.toggle('is-active', index === current && data.status !== 'success' && !repaired);
      item.classList.toggle('repair-on', repaired);
      if (item.classList.contains('is-active')) item.setAttribute('aria-current', 'step');
      else item.removeAttribute('aria-current');
      const icon = item.querySelector('.step-icon');
      icon.textContent = item.classList.contains('is-done') ? '✓' : String(index + 1);
    });
    const friendly = friendlyStep[step] || 'Working on your app';
    if (data.status === 'success') setStatus((data.replay || data.result?.replay) ? 'Pre-built example ready' : 'App ready', 'done');
    else if (data.status === 'failed') setStatus('Build needs another try', 'error');
    else if (data.status === 'queued' && data.queue_position) setStatus(`In queue · position ${data.queue_position}`, 'busy');
    else setStatus(friendly, 'busy');
  }

  function detectLanguage(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    return ({ py: 'python', js: 'javascript', css: 'css', html: 'html', txt: 'plain' })[ext] || 'plain';
  }

  function escapeHtml(text) {
    return text.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
  }

  async function copyText(value) {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return;
    }
    const temporary = document.createElement('textarea');
    temporary.value = value;
    temporary.setAttribute('readonly', '');
    temporary.style.position = 'fixed';
    temporary.style.opacity = '0';
    document.body.append(temporary);
    temporary.select();
    const copied = document.execCommand('copy');
    temporary.remove();
    if (!copied) throw new Error('Copy unavailable');
  }

  function highlight(source, language) {
    let safe = escapeHtml(source);
    if (language === 'html') {
      safe = safe.replace(/(&lt;\/?[\w-]+)([^&]*?)(&gt;)/g, (_all, tag, attrs, close) => {
        const styledAttrs = attrs.replace(/([\w-]+)(=)(&quot;.*?&quot;|&#39;.*?&#39;)/g, '<span class="token-attr">$1</span>$2<span class="token-string">$3</span>');
        return `<span class="token-tag">${tag}</span>${styledAttrs}<span class="token-tag">${close}</span>`;
      });
      return safe;
    }
    const commentMark = language === 'python' ? '#' : '//';
    const tokenPattern = /(#[^\n]*|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:async|await|def|class|return|if|else|elif|for|while|in|import|from|as|with|try|except|finally|raise|pass|True|False|None|const|let|var|function|new|this|throw|catch|export|default|public|private|static|SELECT|FROM|WHERE|INSERT|INTO|VALUES|CREATE|TABLE|PRIMARY|KEY|NOT|NULL|AND|OR)\b|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_$][\w$]*\b)/g;
    return safe.replace(tokenPattern, (token) => {
      if (token.startsWith(commentMark)) return `<span class="token-comment">${token}</span>`;
      if (/^("|')/.test(token)) return `<span class="token-string">${token}</span>`;
      if (/^\d/.test(token)) return `<span class="token-number">${token}</span>`;
      if (/^(SELECT|FROM|WHERE|INSERT|INTO|VALUES|CREATE|TABLE|PRIMARY|KEY|NOT|NULL|AND|OR)$/.test(token)) return `<span class="token-keyword">${token}</span>`;
      if (/^(async|await|def|class|return|if|else|elif|for|while|in|import|from|as|with|try|except|finally|raise|pass|True|False|None|const|let|var|function|new|this|throw|catch|export|default|public|private|static)$/.test(token)) return `<span class="token-keyword">${token}</span>`;
      return `<span class="token-function">${token}</span>`;
    });
  }

  function revealCode(source, filename) {
    const token = ++revealToken;
    const language = detectLanguage(filename);
    codeOutput.className = `language-${language}`;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced || source.length < 160) {
      codeOutput.innerHTML = highlight(source, language);
      return;
    }
    codeOutput.textContent = '';
    let cursor = 0;
    const chunk = 90;
    const writeNext = () => {
      if (token !== revealToken) return;
      cursor = Math.min(cursor + chunk, source.length);
      codeOutput.textContent = source.slice(0, cursor);
      if (cursor < source.length) {
        window.setTimeout(writeNext, 14);
      } else {
        codeOutput.innerHTML = highlight(source, language);
      }
    };
    writeNext();
  }

  function selectFile(filename) {
    activeFile = filename;
    $$('#file-tabs .file-tab').forEach((tab) => tab.setAttribute('aria-selected', String(tab.dataset.file === filename)));
    const file = fileByName.get(filename);
    if (!file) return;
    activeFilename.textContent = filename;
    $('#copy-code').disabled = false;
    revealCode(file.content, filename);
  }

  fileTabs.addEventListener('keydown', (event) => {
    const tabs = [...fileTabs.querySelectorAll('.file-tab')];
    const index = tabs.indexOf(document.activeElement);
    if (index < 0 || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
      : (index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
    tabs[next]?.focus();
    selectFile(tabs[next]?.dataset.file);
  });

  function renderFiles(files) {
    if (!Array.isArray(files)) return;
    const snapshot = JSON.stringify(files.map((file) => [file.filename, file.content]));
    if (snapshot === lastFileSnapshot) return;
    lastFileSnapshot = snapshot;
    const previous = activeFile;
    fileByName.clear();
    files.forEach((file) => fileByName.set(file.filename, file));
    const wasEmpty = fileTabs.querySelector('.file-empty');
    if (wasEmpty) fileTabs.replaceChildren();
    for (const file of files) {
      let tab = [...fileTabs.querySelectorAll('.file-tab')].find((item) => item.dataset.file === file.filename);
      if (!tab) {
        tab = document.createElement('button');
        tab.type = 'button';
        tab.className = 'file-tab';
        tab.setAttribute('role', 'tab');
        tab.setAttribute('aria-controls', 'code-output');
        tab.dataset.file = file.filename;
        tab.textContent = file.filename;
        tab.addEventListener('click', () => selectFile(file.filename));
        fileTabs.append(tab);
      }
    }
    for (const tab of [...fileTabs.querySelectorAll('.file-tab')]) {
      if (!fileByName.has(tab.dataset.file)) tab.remove();
    }
    $('#file-count').textContent = `${files.length} ${files.length === 1 ? 'file' : 'files'}`;
    const toSelect = previous && fileByName.has(previous) ? previous : files[files.length - 1]?.filename;
    if (toSelect) selectFile(toSelect);
  }

  function renderBlueprint(spec) {
    const host = $('#blueprint-content');
    host.replaceChildren();
    const entities = Array.isArray(spec?.entities) ? spec.entities : [];
    const relationships = Array.isArray(spec?.relationships) ? spec.relationships : [];
    if (!entities.length) {
      const empty = document.createElement('div');
      empty.className = 'blueprint-empty';
      empty.textContent = 'No data entities were needed for this app.';
      host.append(empty);
      return;
    }
    const list = document.createElement('div');
    list.className = 'entity-list';
    entities.forEach((entity) => {
      const card = document.createElement('article');
      card.className = 'entity-card';
      const heading = document.createElement('div');
      heading.className = 'entity-head';
      const title = document.createElement('span');
      title.textContent = entity.name || 'Entity';
      const table = document.createElement('span');
      table.className = 'entity-table';
      table.textContent = (entity.name || 'entity').toLowerCase().replace(/\W+/g, '_');
      heading.append(title, table);
      const fields = document.createElement('div');
      fields.className = 'entity-fields';
      const idPill = document.createElement('span');
      idPill.className = 'field-pill primary';
      idPill.textContent = 'id · primary key';
      fields.append(idPill);
      (entity.fields || []).forEach((field) => {
        const pill = document.createElement('span');
        pill.className = 'field-pill';
        pill.textContent = `${field.name || 'field'} · ${field.type || 'text'}`;
        fields.append(pill);
      });
      card.append(heading, fields);
      list.append(card);
    });
    host.append(list);
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 600 260');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Entity relationship diagram');
    svg.classList.add('er-diagram');
    const count = entities.length;
    const boxWidth = Math.min(140, Math.max(94, 500 / Math.max(count, 1)));
    const gap = count > 1 ? (500 - count * boxWidth) / (count - 1) : 0;
    const positions = new Map();
    entities.forEach((entity, index) => {
      const x = 50 + index * (boxWidth + gap);
      positions.set(entity.name, { x, center: x + boxWidth / 2 });
    });
    relationships.forEach((relation) => {
      const from = positions.get(relation.from);
      const to = positions.get(relation.to);
      if (!from || !to) return;
      const line = document.createElementNS(svg.namespaceURI, 'line');
      line.setAttribute('x1', String(from.center)); line.setAttribute('y1', '106');
      line.setAttribute('x2', String(to.center)); line.setAttribute('y2', '106');
      svg.append(line);
      const label = document.createElementNS(svg.namespaceURI, 'text');
      label.setAttribute('x', String((from.center + to.center) / 2)); label.setAttribute('y', '96'); label.setAttribute('text-anchor', 'middle');
      label.textContent = relation.type || 'related';
      svg.append(label);
    });
    entities.forEach((entity) => {
      const point = positions.get(entity.name);
      const rect = document.createElementNS(svg.namespaceURI, 'rect');
      rect.setAttribute('x', String(point.x)); rect.setAttribute('y', '108'); rect.setAttribute('width', String(boxWidth)); rect.setAttribute('height', '52'); rect.setAttribute('rx', '7');
      const title = document.createElementNS(svg.namespaceURI, 'text');
      title.setAttribute('x', String(point.center)); title.setAttribute('y', '139'); title.setAttribute('text-anchor', 'middle'); title.classList.add('er-label'); title.textContent = entity.name || 'Entity';
      svg.append(rect, title);
    });
    host.append(svg);
  }

  function showBlueprint(show) {
    $('#workspace-grid').classList.toggle('hidden', show);
    $('#blueprint-panel').classList.toggle('hidden', !show);
    $('#blueprint-tab').setAttribute('aria-expanded', String(show));
    $('#blueprint-tab').innerHTML = show ? 'Blueprint <span aria-hidden="true">⌃</span>' : 'Blueprint <span aria-hidden="true">⌄</span>';
    if (show) $('#blueprint-title').focus?.();
  }

  function showStats(data) {
    const allFiles = fileByName.size;
    const lines = [...fileByName.values()].reduce((sum, file) => sum + (file.content.match(/\n/g) || []).length + 1, 0);
    const elapsed = Math.max(1, Math.round(Number(data.duration_seconds) || ((performance.now() - startedAt) / 1000)));
    buildSeconds = elapsed;
    const verify = data.verification?.verification || data.result?.verification?.verification || {};
    const repairList = data.verification?.repairs || data.result?.verification?.repairs || [];
    $('#stat-time').textContent = elapsed < 60 ? `${elapsed}s` : `${Math.floor(elapsed / 60)}m ${elapsed % 60}s`;
    $('#stat-files').textContent = String(allFiles);
    $('#stat-lines').textContent = lines.toLocaleString();
    $('#stat-endpoints').textContent = String(verify.endpoints_verified ?? verify.routes_checked?.length ?? '—');
    $('#stat-fixes').textContent = String(repairList.length);
    $('#stats-strip').classList.remove('hidden');
    const checkNames = {
      create_and_persist: 'Create record and confirm SQLite persistence',
      update: 'Update record',
      read_after_update: 'Read updated record',
      delete: 'Delete record',
      read_after_delete: 'Confirm deleted record is missing',
      missing_record: 'Missing-record route returns 404',
    };
    const checks = $('#verification-checks');
    checks.replaceChildren();
    (verify.workflow_checks || []).forEach((item) => {
      const row = document.createElement('li');
      row.textContent = checkNames[item.operation] || item.operation.replaceAll('_', ' ');
      checks.append(row);
    });
    if (!checks.children.length) {
      if (Number(verify.endpoints_verified) > 0) {
        const row = document.createElement('li');
        row.textContent = `${verify.endpoints_verified} API endpoint checks passed`;
        checks.append(row);
      }
      if (verify.records_persisted) {
        const row = document.createElement('li');
        row.textContent = 'Temporary verification record persisted in SQLite';
        checks.append(row);
      }
    }
    if (verify.status === 'ok' && checks.children.length) $('#verification-summary').classList.remove('hidden');
    $('#compare-panel').classList.remove('hidden');
    $('#compare-time').textContent = `${elapsed}s`;
    $('#modify-panel').classList.remove('hidden');
    refreshVersionControls(data.version_count ?? 1);
    if (repairList.length) {
      const diff = $('#repair-diff');
      diff.replaceChildren();
      repairList.forEach((repair) => {
        const file = document.createElement('div');
        file.textContent = repair.file;
        const before = document.createElement('div');
        before.className = 'diff-before';
        before.textContent = `− Found: ${repair.before}`;
        const after = document.createElement('div');
        after.className = 'diff-after';
        after.textContent = `+ Fixed: ${repair.after}`;
        diff.append(file, before, after);
      });
      $('#repair-summary').classList.remove('hidden');
    }
  }

  function refreshVersionControls(count) {
    const versions = Math.max(1, Number(count) || 1);
    $('#version-count').textContent = `Version ${versions}`;
    $('#version-history').textContent = versions > 1 ? `${versions - 1} earlier ${versions === 2 ? 'version' : 'versions'} saved` : 'No earlier versions';
    $('#undo-button').disabled = versions <= 1;
  }

  function renderHistory(data) {
    const host = $('#history-list');
    host.replaceChildren();
    (data.jobs || []).slice(0, 8).forEach((job) => {
      const card = document.createElement('article');
      card.className = 'history-card'; card.tabIndex = 0; card.setAttribute('role', 'link');
      const thumb = document.createElement('div'); thumb.className = 'history-thumb'; thumb.setAttribute('aria-hidden', 'true');
      const title = document.createElement('strong'); title.textContent = job.app_name || 'Generated app';
      const prompt = document.createElement('p'); prompt.textContent = job.prompt || 'Verified ForgeAI build';
      const time = document.createElement('span'); time.textContent = job.created_at ? new Date(job.created_at).toLocaleString() : 'Recent build';
      const source = document.createElement('span'); source.className = 'history-source';
      source.textContent = job.replay ? 'Stored showcase replay · verified' : 'AI-generated build · verified';
      card.append(thumb, title, source, prompt, time);
      const open = () => reopenBuild(job.job_id);
      card.addEventListener('click', open);
      card.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); } });
      host.append(card);
    });
    if (!host.children.length) { const empty = document.createElement('p'); empty.className = 'gallery-empty'; empty.textContent = 'Your verified builds will appear here.'; host.append(empty); }
    $('#apps-forged').textContent = data.apps_forged ?? 0;
  }

  async function reopenBuild(jobId) {
    try {
      const response = await fetch(`/status/${encodeURIComponent(jobId)}`);
      const data = await response.json();
      if (!response.ok || data.status !== 'success') return;
      if (pollTimer) clearInterval(pollTimer);
      currentJobId = jobId; buildSection.classList.remove('hidden'); resetWorkspace();
      renderFiles(data.generated_files || []); renderBlueprint(data.spec || {});
      startedAt = performance.now(); buildSeconds = Number(data.duration_seconds) || 1;
      applyJobUpdate({ ...data, job_id: jobId });
      buildSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      $('#job-label').textContent = `Build ${jobId.slice(0, 8)}`;
    } catch (_error) { /* The gallery stays usable during brief local network interruptions. */ }
  }

  async function pollModification(taskId) {
    const button = $('#modify-submit'); button.disabled = true;
    $('#modify-message').textContent = 'Updating selected files and verifying your app…';
    try {
      while (true) {
        const response = await fetch(`/status/${encodeURIComponent(taskId)}`);
        const task = await response.json();
        if (task.status === 'success') {
          const parentResponse = await fetch(`/status/${encodeURIComponent(currentJobId)}`);
          const parent = await parentResponse.json();
          renderFiles(parent.generated_files || []); renderBlueprint(parent.spec || {});
          previewFrame.src = `${previewUrl}?refresh=${Date.now()}`;
          refreshVersionControls(parent.version_count);
          $('#modify-message').textContent = `Change verified. Updated: ${(task.result?.modified_files || []).join(', ')}.`;
          refreshHistory(); return;
        }
        if (task.status === 'failed') { $('#modify-message').textContent = task.result?.reason || 'That change did not pass verification. Your earlier version is still running.'; return; }
        await new Promise((resolve) => setTimeout(resolve, 900));
      }
    } catch (_error) { $('#modify-message').textContent = 'Could not reach ForgeAI while applying the change. Try again.'; }
    finally { button.disabled = false; }
  }

  function refreshHistory() {
    fetch('/history').then((response) => response.json()).then(renderHistory).catch(() => {});
  }

  function displayError(data) {
    generateButton.disabled = false;
    generateButton.querySelector('span').textContent = 'Generate app';
    const reason = data.result?.reason || data.verification?.error || 'ForgeAI could not complete this build yet.';
    let plain = 'ForgeAI couldn’t finish this app. Your idea is still here—try again in a moment.';
    if (/rate limit|429|quota/i.test(reason)) plain = 'The AI service is busy right now. Wait a moment, then try again.';
    else if (/api key|authentication|forbidden/i.test(reason)) plain = 'The AI service is not connected. Ask the project team to check its setup.';
    else if (/timed out|timeout/i.test(reason)) plain = 'The app took too long to start. Try the build again.';
    $('#error-message').textContent = plain;
    const details = data.result?.details || data.verification || {};
    $('#developer-error').textContent = [reason, details.error, details.traceback, details.verification?.traceback].filter(Boolean).join('\n\n');
    $('#error-panel').classList.remove('hidden');
    setStatus('Build needs another try', 'error');
  }

  function finishSuccess(data) {
    const jobId = data.job_id || currentJobId;
    previewUrl = new URL(`/preview/${encodeURIComponent(jobId)}/`, window.location.origin).href;
    shareUrl = previewUrl;
    if (['localhost', '127.0.0.1', '::1'].includes(window.location.hostname)) {
      shareUrlReady = fetch('/network-address').then((response) => response.json()).then((network) => {
        if (network.origin) shareUrl = new URL(`/preview/${encodeURIComponent(jobId)}/`, network.origin).href;
      }).catch(() => {});
    }
    previewFrame.src = previewUrl;
    $('#preview-address').textContent = previewUrl;
    $('#preview-state').textContent = 'Your app is running';
    $('#open-preview').href = previewUrl;
    previewPlaceholder.classList.add('hidden');
    $('#result-actions').classList.remove('hidden');
    if (data.replay || data.result?.replay) {
      $('.result-intro strong').textContent = data.result?.label || 'Prebuilt showcase replay';
      $('.result-intro > div > span').textContent = 'Loaded from ForgeAI’s local showcase catalog; no AI generation was used.';
      const banner = $('#live-failure-banner');
      if (data.result?.live_generation_failure) {
        banner.textContent = `Live generation failed: ${data.result.live_generation_failure}. Showing a pre-built example instead.`;
        banner.classList.remove('hidden');
      } else banner.classList.add('hidden');
    } else {
      $('#live-failure-banner').classList.add('hidden');
      $('.result-intro strong').textContent = 'Your app is ready.';
      $('.result-intro > div > span').textContent = 'It passed the verification checks shown below and is running locally.';
    }
    showStats(data);
    fetch('/history').then((response) => response.json()).then((history) => { $('#apps-forged').textContent = history.apps_forged ?? 0; }).catch(() => {});
  }

  function applyJobUpdate(data) {
    const logs = Array.isArray(data.logs) ? data.logs : [];
    for (const message of logs.slice(renderedLogCount)) appendLog(message);
    renderedLogCount = logs.length;
    if (data.current_step) $('#job-label').textContent = friendlyStep[data.current_step] || 'Build in progress';
    renderPipeline(data);
    renderFiles(data.generated_files || []);
    if (data.spec) renderBlueprint(data.spec);
    if (data.status === 'success' && data.result?.success) {
      if (pollTimer) window.clearInterval(pollTimer);
      pollTimer = null;
      finishSuccess(data);
      generateButton.disabled = false;
      generateButton.querySelector('span').textContent = 'Generate app';
      return true;
    }
    if (data.status === 'failed' || data.result?.success === false) {
      if (pollTimer) window.clearInterval(pollTimer);
      pollTimer = null;
      displayError(data);
      return true;
    }
    return false;
  }

  async function pollOnce(jobId) {
    if (pollInProgress) return;
    pollInProgress = true;
    try {
      const response = await fetch(`/status/${encodeURIComponent(jobId)}`, { headers: { Accept: 'application/json' } });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Status unavailable');
      consecutivePollFailures = 0;
      return applyJobUpdate({ ...data, job_id: jobId });
    } catch (_error) {
      // Keep polling during brief network interruptions; raw errors belong in developer details only.
      consecutivePollFailures += 1;
      if (consecutivePollFailures >= 10) {
        if (pollTimer) window.clearInterval(pollTimer);
        pollTimer = null;
        displayError({ result: { reason: 'ForgeAI lost the connection while checking this build. It may still be running; you can try again or use offline replay.' } });
        appendLog('Status connection was lost. The loading state has been stopped.');
        return true;
      }
      return false;
    } finally {
      pollInProgress = false;
    }
  }

  async function startBuild(prompt) {
    if (generateButton.disabled) return;
    if (pollTimer) window.clearInterval(pollTimer);
    buildSection.classList.remove('hidden');
    buildSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    resetWorkspace();
    currentJobId = null;
    setStatus('Sending your idea', 'busy');
    $('#job-label').textContent = 'Creating a build job';
    startedAt = performance.now();
    generateButton.disabled = true;
    generateButton.querySelector('span').textContent = 'Building…';
    try {
      const response = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ prompt, demo_mode: $('#demo-mode').checked })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Request could not be started');
      currentJobId = data.job_id;
      appendLog('Build queued. ForgeAI is getting to work.');
      setStatus('In the queue', 'busy');
      $('#job-label').textContent = `Build ${data.job_id.slice(0, 8)}`;
      const terminal = await pollOnce(currentJobId);
      if (!terminal && !pollTimer) pollTimer = window.setInterval(() => pollOnce(currentJobId), 1200);
    } catch (error) {
      const networkFailure = error instanceof TypeError;
      const reason = networkFailure
        ? 'ForgeAI could not reach the local server. Make sure the backend is running.'
        : (error?.message || 'ForgeAI could not start this build.');
      displayError({ result: { reason } });
      if (networkFailure) appendLog('Could not reach ForgeAI. Check that the local server is running.');
      else appendLog(`Build request failed: ${reason}`);
    } finally {
      if (!currentJobId) {
        generateButton.disabled = false;
        generateButton.querySelector('span').textContent = 'Generate app';
      }
    }
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const prompt = promptInput.value.trim();
    if (prompt) startBuild(prompt);
  });

  $$('.example-chip').forEach((chip) => chip.addEventListener('click', () => {
    promptInput.value = chip.dataset.prompt || '';
    promptInput.focus();
    promptInput.dispatchEvent(new Event('input', { bubbles: true }));
  }));

  $('#theme-toggle').addEventListener('click', () => {
    const root = document.documentElement;
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    localStorage.setItem('forgeai-theme', next);
    $('#theme-toggle').setAttribute('aria-label', `Switch to ${next === 'dark' ? 'light' : 'dark'} theme`);
  });
  const savedTheme = localStorage.getItem('forgeai-theme');
  if (savedTheme === 'light' || savedTheme === 'dark') document.documentElement.dataset.theme = savedTheme;
  $('#theme-toggle').setAttribute('aria-label', `Switch to ${document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'} theme`);

  $('#blueprint-tab').addEventListener('click', () => showBlueprint($('#blueprint-tab').getAttribute('aria-expanded') !== 'true'));
  $('#close-blueprint').addEventListener('click', () => showBlueprint(false));
  $$('.width-switch button').forEach((button) => button.addEventListener('click', () => {
    const phone = button.dataset.width === 'phone';
    iframeWrap.classList.toggle('phone-width', phone);
    $$('.width-switch button').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
  }));
  $('#refresh-preview').addEventListener('click', () => { if (previewUrl) previewFrame.src = `${previewUrl}?refresh=${Date.now()}`; });
  $('#copy-code').addEventListener('click', async () => {
    const file = fileByName.get(activeFile);
    if (!file) return;
    try { await copyText(file.content); $('#copy-code').textContent = 'Copied'; window.setTimeout(() => { $('#copy-code').textContent = 'Copy code'; }, 1200); } catch (_error) { $('#copy-code').textContent = 'Copy unavailable'; }
  });
  $('#download-btn').addEventListener('click', async () => {
    if (!currentJobId) return;
    try {
      const response = await fetch(`/download/${encodeURIComponent(currentJobId)}`);
      if (!response.ok) throw new Error('Download unavailable');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url; anchor.download = `forgeai-${currentJobId.slice(0, 8)}.zip`; anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (_error) { appendLog('The download could not be prepared. Try again in a moment.'); }
  });
  $('#copy-share').addEventListener('click', async () => {
    if (!shareUrl) return;
    try { await shareUrlReady; await copyText(shareUrl); $('#copy-share').innerHTML = 'Link copied <span aria-hidden="true">✓</span>'; window.setTimeout(() => { $('#copy-share').innerHTML = 'Copy share link <span aria-hidden="true">⧉</span>'; }, 1400); } catch (_error) { appendLog('Copy is unavailable in this browser. Select the preview address to share it.'); }
  });
  $('#show-qr').addEventListener('click', async () => {
    const panel = $('#qr-panel');
    const show = panel.classList.contains('hidden');
    panel.classList.toggle('hidden', !show);
    if (show && previewUrl && window.ForgeQR) {
      try { await shareUrlReady; $('#qr-code').innerHTML = window.ForgeQR.toSvg(shareUrl || previewUrl); }
      catch (_error) { $('#qr-code').textContent = 'Use the copied link to open this preview.'; }
    }
  });
  $('#retry-btn').addEventListener('click', () => form.requestSubmit());
  $('#offline-replay-btn').addEventListener('click', () => {
    $('#demo-mode').checked = true;
    form.requestSubmit();
  });

  const templates = [
    ['Productivity', 'Weekly task planner', 'A weekly planner to organize tasks by day, priority, and completion status.'],
    ['Productivity', 'Study session tracker', 'A study tracker to log subjects, focus sessions, durations, and weekly goals.'],
    ['Health', 'Workout journal', 'A workout journal to save exercises, sets, duration, and progress over time.'],
    ['Health', 'Meal planner', 'A meal planner to save meals, ingredients, dietary tags, and a weekly plan.'],
    ['Education', 'Class attendance', 'A class attendance register to record students, dates, and present or absent status.'],
    ['Education', 'Flashcard library', 'A flashcard app to create study decks, review cards, and track what I know.'],
    ['Business', 'Stock manager', 'An inventory manager to track products, categories, quantities, and low-stock items.'],
    ['Business', 'Expense splitter', 'An expense splitter to track shared purchases, who paid, participants, and balances.']
  ];
  const templateHost = $('#template-list');
  templates.forEach(([category, title, prompt]) => {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'template-card';
    const label = document.createElement('span'); label.textContent = category;
    const name = document.createElement('strong'); name.textContent = title;
    button.append(label, name); button.addEventListener('click', () => { promptInput.value = prompt; promptInput.focus(); }); templateHost.append(button);
  });

  $('#modify-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!currentJobId) return;
    const change = $('#modify-input').value.trim(); if (!change) return;
    $('#modify-submit').disabled = true;
    try {
      const response = await fetch(`/modify/${encodeURIComponent(currentJobId)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ change }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Could not start the update');
      $('#modify-input').value = '';
      pollModification(payload.job_id);
    } catch (error) { $('#modify-message').textContent = error.message || 'Could not start that change.'; $('#modify-submit').disabled = false; }
  });
  $$('.modify-suggestions button').forEach((button) => button.addEventListener('click', () => { $('#modify-input').value = button.dataset.change; $('#modify-input').focus(); }));
  $('#undo-button').addEventListener('click', async () => {
    if (!currentJobId) return;
    const button = $('#undo-button'); button.disabled = true; $('#modify-message').textContent = 'Restoring and verifying the earlier version…';
    try {
      const response = await fetch(`/undo/${encodeURIComponent(currentJobId)}`, { method: 'POST' }); const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Undo failed');
      renderFiles(data.generated_files || []); renderBlueprint(data.spec || {}); previewFrame.src = `${previewUrl}?refresh=${Date.now()}`;
      refreshVersionControls(data.version_count); $('#modify-message').textContent = 'Earlier version restored and verified.';
    } catch (error) { $('#modify-message').textContent = error.message || 'Could not restore that version.'; button.disabled = false; }
  });
  $('#compare-toggle').addEventListener('change', (event) => $('#compare-content').classList.toggle('hidden', !event.target.checked));

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    const voice = $('#voice-button'); voice.classList.remove('hidden');
    voice.addEventListener('click', () => {
      const recognition = new SpeechRecognition(); recognition.lang = navigator.language || 'en-US'; recognition.interimResults = false;
      voice.disabled = true; voice.textContent = 'Listening…';
      recognition.onresult = (event) => { promptInput.value = event.results[0][0].transcript; promptInput.focus(); };
      recognition.onerror = () => { voice.textContent = 'Try speaking again'; };
      recognition.onend = () => { voice.disabled = false; if (voice.textContent === 'Listening…') voice.textContent = 'Speak'; };
      recognition.start();
    });
  }

  refreshHistory();
})();
