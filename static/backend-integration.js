/**
 * TalentSync AI — Backend Integration Layer
 * Drop-in replacement for direct Anthropic calls.
 * Routes through the FastAPI backend for persistence, analytics, and secure API key management.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

const BACKEND_URL = window.BACKEND_URL || 'http://localhost:8000';
const USE_BACKEND = window.USE_BACKEND !== false; // set to false to use direct Anthropic calls

// ─────────────────────────────────────────────────────────────────────────────
// API helpers
// ─────────────────────────────────────────────────────────────────────────────

async function backendRequest(path, options = {}) {
  const url = `${BACKEND_URL}${path}`;
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
  return data;
}

// Override callClaude — proxy through backend so API key stays server-side
async function callClaude(prompt, systemPrompt = '') {
  STATE.aiCalls++;
  updateSidebarStats();
  addLog('AI', 'Calling Claude via backend proxy…');

  if (USE_BACKEND) {
    const body = {
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1000,
      messages: [{ role: 'user', content: prompt }],
    };
    if (systemPrompt) body.system = systemPrompt;

    const data = await backendRequest('/api/claude/proxy', {
      method: 'POST',
      body: JSON.stringify(body),
    });
    addLog('AI', `Claude responded (${data.usage?.output_tokens || '?'} tokens)`);
    return data.content[0]?.text || '';
  } else {
    // Fallback: direct Anthropic call (requires CORS workaround / artifact runner)
    const body = {
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1000,
      messages: [{ role: 'user', content: prompt }],
    };
    if (systemPrompt) body.system = systemPrompt;
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || 'API error');
    addLog('AI', `Claude responded (${data.usage?.output_tokens || '?'} tokens)`);
    return data.content[0]?.text || '';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline — use backend endpoint (stores results in DB)
// ─────────────────────────────────────────────────────────────────────────────

async function runFullPipelineBackend(resumeText, filename = 'resume') {
  const startTime = Date.now();
  clearPipelineLog();

  // Show agents active immediately
  setAgentState('parser', 'active');
  setPipelineMsg(`Running 3-agent pipeline on ${filename}…`);
  showNotif('Running AI pipeline via backend…');

  const jd = document.getElementById('jd-textarea')?.value || '';

  const formData = new FormData();
  formData.append('resume_text', resumeText);
  formData.append('filename', filename);
  if (jd) formData.append('job_description', jd);

  try {
    addPipelineLog('parser', `[PARSER] Starting extraction on ${filename}`);
    
    const result = await fetch(`${BACKEND_URL}/api/pipeline/run`, {
      method: 'POST',
      body: formData,
    }).then(async r => {
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `Pipeline error ${r.status}`);
      }
      return r.json();
    });

    // Replay pipeline logs from backend response
    (result.pipeline_logs || []).forEach(log => {
      setAgentState(log.agent, 'active');
      addPipelineLog(log.agent, log.message);
    });

    setAgentState('parser', 'done');
    setAgentState('normalizer', 'done');
    setAgentState('matcher', 'done');
    setAgentState('output', 'done');

    // Update STATE
    STATE.currentCandidate = result.parsed;
    STATE.parsed++;
    STATE.currentCandidateId = result.candidate_id;
    STATE.skillsExtracted.push(...(result.normalized_skills || []));
    STATE.skillNormalizations.push(...(result.normalized_skills || []));
    STATE.matchScores.push(result.match?.match_score || 0);

    // Update UI
    updateCandidateProfile(result.parsed);
    updateMatchScore(result.match);
    updateConfidenceBars(result.parsed.confidence || { skills: 95, experience: 92, education: 98 });
    updateSkillTable(result.normalized_skills || []);
    updateDashboard();
    updateCurrentCandidateInfo(result.parsed);

    // Hero stats
    const totalMs = result.pipeline_time_ms || (Date.now() - startTime);
    document.getElementById('stat-accuracy').textContent = `${result.match?.match_score || 0}%`;
    document.getElementById('stat-skills').textContent = result.normalized_skills?.length || 0;
    document.getElementById('stat-time').textContent = `${(totalMs / 1000).toFixed(1)}s`;
    setPipelineMsg(`Pipeline complete in ${totalMs}ms — ${result.parsed?.skills?.length || 0} skills, ${result.match?.match_score || 0}% match`);
    showNotif(`✓ Pipeline complete! ${result.parsed?.name} scored ${result.match?.match_score}%`);
    updateSidebarStats();

    // Load candidate list
    await loadCandidateList();

    addPipelineLog('output', `[OUTPUT] ✓ Saved as candidate ${result.candidate_id}`);

  } catch (e) {
    showNotif(`Pipeline error: ${e.message}`);
    addPipelineLog('output', `[OUTPUT] ✗ ${e.message}`);
    console.error('Pipeline error:', e);
    // Fall back to local pipeline
    addLog('INFO', 'Falling back to direct AI pipeline…');
    await runFullPipelineDirect(resumeText, filename);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Candidate List (persistent across sessions)
// ─────────────────────────────────────────────────────────────────────────────

async function loadCandidateList() {
  try {
    const data = await backendRequest('/api/candidates?limit=20');
    renderCandidateList(data.candidates || []);
  } catch (e) {
    console.warn('Could not load candidate list:', e);
  }
}

function renderCandidateList(candidates) {
  const el = document.getElementById('candidate-list');
  if (!el) return;
  if (candidates.length === 0) {
    el.innerHTML = '<div style="font-size:12px;color:var(--muted);text-align:center;padding:20px">No candidates yet. Upload a resume to get started.</div>';
    return;
  }
  const verdictColor = { 'Strong Match': 'tag-green', 'Good Match': 'tag-cyan', 'Moderate Match': 'tag-amber', 'Weak Match': 'tag-violet' };
  el.innerHTML = candidates.map(c => `
    <div class="candidate-card" onclick="loadCandidate('${c.id}')" id="cand-card-${c.id}">
      <div style="display:flex;align-items:center;gap:10px">
        <div class="avatar-placeholder" style="width:36px;height:36px;background:linear-gradient(135deg,#00F5FF,#7C3AED);font-size:12px;color:white">
          ${(c.name || '?').split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)}
        </div>
        <div style="flex:1;min-width:0">
          <div style="font-size:12px;font-weight:600;color:white;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${c.name || 'Unknown'}</div>
          <div style="font-size:10px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${c.title || '—'}</div>
        </div>
        <div style="text-align:right;flex-shrink:0">
          <div style="font-size:14px;font-weight:700;color:${c.match_score >= 80 ? 'var(--green)' : c.match_score >= 65 ? 'var(--cyan)' : 'var(--amber)'}">${c.match_score || 0}%</div>
          <div style="font-size:9px;color:var(--muted)">${c.verdict || ''}</div>
        </div>
      </div>
      <div style="margin-top:8px">
        <div class="match-track"><div class="match-fill" data-target="${c.match_score || 0}" style="width:${c.match_score || 0}%"></div></div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px">
        <span style="font-size:9px;color:var(--dimmed)">${new Date(c.created_at).toLocaleDateString()}</span>
        <button onclick="event.stopPropagation(); deleteCandidate('${c.id}')" style="background:none;border:none;color:var(--dimmed);cursor:pointer;font-size:10px;padding:2px 6px;border-radius:4px" onmouseover="this.style.color='#ef4444'" onmouseout="this.style.color='var(--dimmed)'">✕</button>
      </div>
    </div>
  `).join('');
}

async function loadCandidate(candidateId) {
  try {
    const c = await backendRequest(`/api/candidates/${candidateId}`);
    STATE.currentCandidate = c.parsed;
    STATE.currentCandidateId = candidateId;
    updateCandidateProfile(c.parsed);
    updateMatchScore(c.match_data);
    updateSkillTable(c.normalized_skills || []);
    updateCurrentCandidateInfo(c.parsed);
    document.querySelectorAll('.candidate-card').forEach(el => el.classList.remove('selected'));
    document.getElementById(`cand-card-${candidateId}`)?.classList.add('selected');
    document.getElementById('resume-section')?.scrollIntoView({ behavior: 'smooth' });
    showNotif(`Loaded: ${c.name}`);
  } catch (e) {
    showNotif('Could not load candidate');
  }
}

async function deleteCandidate(candidateId) {
  if (!confirm('Delete this candidate?')) return;
  try {
    await backendRequest(`/api/candidates/${candidateId}`, { method: 'DELETE' });
    await loadCandidateList();
    showNotif('Candidate deleted');
  } catch (e) {
    showNotif('Delete failed');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytics — load from backend on page load
// ─────────────────────────────────────────────────────────────────────────────

async function loadDashboardAnalytics() {
  try {
    const data = await backendRequest('/api/analytics/dashboard');
    // Sync STATE with backend data
    if (data.total_candidates > STATE.parsed) {
      STATE.parsed = data.total_candidates;
      document.getElementById('kpi-parsed').textContent = data.total_candidates;
      document.getElementById('parsed-badge').textContent = data.total_candidates + ' total';
    }
    if (data.avg_match_score > 0) {
      document.getElementById('kpi-score').textContent = data.avg_match_score + '%';
    }
    if (data.total_candidates > 0) {
      document.getElementById('kpi-skills').textContent = data.unique_skills;
      document.getElementById('skills-badge').textContent = data.unique_skills + ' unique';
    }

    // Update charts with persistent data
    if (matchChartInst && data.score_buckets) {
      const buckets = data.score_buckets;
      matchChartInst.data.datasets[0].data = [
        buckets['<60'] || 0, buckets['60-70'] || 0, buckets['70-80'] || 0,
        buckets['80-90'] || 0, buckets['90+'] || 0
      ];
      matchChartInst.update();
      document.getElementById('chart-subtitle').textContent = `${data.total_candidates} candidate${data.total_candidates !== 1 ? 's' : ''}`;
    }

    if (skillChartInst && data.skill_categories) {
      const cats = data.skill_categories;
      skillChartInst.data.datasets[0].data = [
        cats['AI/ML'] || 0, cats['Backend'] || 0, cats['Frontend'] || 0,
        cats['DevOps'] || 0, cats['Other'] || 0,
      ];
      skillChartInst.update();
    }

    // Render leaderboard
    renderLeaderboard(data.recent_candidates || []);

    addLog('SUCCESS', `Dashboard synced — ${data.total_candidates} candidates in database`);
  } catch (e) {
    console.warn('Backend offline — running in standalone mode');
    addLog('INFO', 'Backend not reachable — using standalone mode');
  }
}

function renderLeaderboard(candidates) {
  const el = document.getElementById('leaderboard-list');
  if (!el || candidates.length === 0) return;
  el.innerHTML = candidates.slice(0, 5).map((c, i) => `
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)">
      <div style="width:20px;text-align:center;font-size:11px;font-weight:700;color:${i === 0 ? 'var(--amber)' : 'var(--muted)'}">${i + 1}</div>
      <div style="flex:1">
        <div style="font-size:12px;font-weight:600;color:white">${c.name || 'Unknown'}</div>
        <div style="font-size:10px;color:var(--muted)">${c.title || ''}</div>
      </div>
      <div style="font-size:14px;font-weight:700;color:${c.match_score >= 80 ? 'var(--green)' : 'var(--cyan)'}">${c.match_score || 0}%</div>
    </div>
  `).join('');
}

// ─────────────────────────────────────────────────────────────────────────────
// Skill report via backend
// ─────────────────────────────────────────────────────────────────────────────

async function generateSkillReportBackend() {
  if (STATE.skillNormalizations.length === 0) {
    showNotif('No skills to analyze. Parse a resume first.');
    return;
  }
  showNotif('Generating AI skill report…');
  try {
    const data = await backendRequest('/api/skills/report', {
      method: 'POST',
      body: JSON.stringify({ skills: STATE.skillNormalizations }),
    });
    document.getElementById('skill-insights').innerHTML = `
      <div style="padding:14px;border-radius:8px;background:rgba(124,58,237,.05);border:1px solid rgba(124,58,237,.15)">
        <div style="font-size:11px;font-weight:700;color:#a78bfa;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px">AI Skill Report</div>
        <div style="font-size:12px;color:#cbd5e1;line-height:1.7;white-space:pre-wrap">${data.report}</div>
      </div>
    `;
    showNotif('Skill report generated!');
  } catch (e) {
    showNotif('Generating report locally…');
    await generateSkillReport();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Semantic match via backend  
// ─────────────────────────────────────────────────────────────────────────────

async function runSemanticMatchBackend() {
  const jd = document.getElementById('jd-textarea')?.value?.trim();
  if (!jd) { showNotif('Please enter a job description first'); return; }
  if (!STATE.currentCandidate) { showNotif('Please parse a resume first'); return; }

  const btn = document.getElementById('match-btn');
  btn.disabled = true;
  btn.textContent = 'Running AI Match…';

  try {
    const data = await backendRequest('/api/match/semantic', {
      method: 'POST',
      body: JSON.stringify({
        candidate: STATE.currentCandidate,
        normalized_skills: STATE.skillNormalizations,
        job_description: jd,
        candidate_id: STATE.currentCandidateId || null,
      }),
    });

    STATE.matchScores.push(data.match_score);
    renderSemanticMatchResult(data);
    updateDashboard();
    showNotif(`Semantic match: ${data.match_score}% — ${data.verdict}`);
  } catch (e) {
    showNotif('Using local match analysis…');
    await runSemanticMatch();
  }

  btn.disabled = false;
  btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> Run Semantic Match with AI`;
}

function renderSemanticMatchResult(data) {
  document.getElementById('match-results').innerHTML = `
    <div class="card" style="padding:18px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <div>
          <div style="font-size:28px;font-weight:800;color:white">${data.match_score}%</div>
          <div style="font-size:13px;color:var(--cyan);font-weight:700">${data.verdict}</div>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px;align-items:flex-end">
          <span class="tag tag-cyan" style="font-size:10px">Seniority: ${data.seniority_fit || '—'}</span>
          <span class="tag tag-violet" style="font-size:10px">Technical: ${data.technical_fit || '—'}</span>
          ${data.hiring_signal ? `<span class="tag ${data.hiring_signal === 'proceed' ? 'tag-green' : data.hiring_signal === 'consider' ? 'tag-amber' : 'tag-violet'}" style="font-size:10px">Signal: ${data.hiring_signal}</span>` : ''}
        </div>
      </div>
      <div class="match-track" style="margin-bottom:16px"><div class="match-fill" id="semantic-bar" style="width:${data.match_score}%"></div></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px">
        <div>
          <div style="font-size:10px;font-weight:700;color:var(--green);margin-bottom:6px;text-transform:uppercase;letter-spacing:1px">Matched Skills</div>
          <div style="display:flex;flex-wrap:wrap;gap:4px">${(data.matched_skills || []).map(s => `<span class="tag tag-green" style="font-size:10px">${s}</span>`).join('')}</div>
        </div>
        <div>
          <div style="font-size:10px;font-weight:700;color:var(--amber);margin-bottom:6px;text-transform:uppercase;letter-spacing:1px">Missing Skills</div>
          <div style="display:flex;flex-wrap:wrap;gap:4px">${(data.missing_skills || []).map(s => `<span class="tag tag-amber" style="font-size:10px">${s}</span>`).join('')}</div>
        </div>
      </div>
      <div style="padding:12px;border-radius:8px;background:rgba(0,245,255,.05);border:1px solid rgba(0,245,255,.1);margin-bottom:12px">
        <div style="font-size:10px;font-weight:700;color:var(--cyan);margin-bottom:4px;text-transform:uppercase">AI Recommendation</div>
        <p style="font-size:12px;color:#cbd5e1;line-height:1.6">${data.recommendation}</p>
      </div>
      ${(data.interview_questions || []).length ? `
      <div style="margin-bottom:12px">
        <div style="font-size:10px;font-weight:700;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:1px">Interview Questions</div>
        <div style="display:flex;flex-direction:column;gap:5px">
          ${data.interview_questions.map((q, i) => `<div style="font-size:11px;color:#cbd5e1;padding:6px 10px;border-radius:6px;background:rgba(255,255,255,.03);border-left:2px solid var(--violet)">${i + 1}. ${q}</div>`).join('')}
        </div>
      </div>` : ''}
      <div style="display:flex;gap:10px">
        <div style="flex:1;padding:8px;border-radius:6px;background:rgba(255,255,255,.03);text-align:center">
          <div style="font-size:11px;color:var(--muted)">Time to Hire</div>
          <div style="font-size:12px;font-weight:600;color:white">${data.time_to_hire_estimate || '—'}</div>
        </div>
        <div style="flex:1;padding:8px;border-radius:6px;background:rgba(255,255,255,.03);text-align:center">
          <div style="font-size:11px;color:var(--muted)">Offer Percentile</div>
          <div style="font-size:12px;font-weight:600;color:white">${data.offer_range_percentile || '—'}</div>
        </div>
      </div>
    </div>
  `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Override runFullPipeline to use backend
// ─────────────────────────────────────────────────────────────────────────────
const runFullPipelineDirect = typeof runFullPipeline === 'function' ? runFullPipeline : null;
window.runFullPipeline = runFullPipelineBackend;

// ─────────────────────────────────────────────────────────────────────────────
// Init — bootstrap backend connection on page load
// ─────────────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  setTimeout(async () => {
    try {
      const health = await fetch(`${BACKEND_URL}/health`).then(r => r.json());
      addLog('SUCCESS', `Backend connected — ${health.candidates_stored} candidates in DB`);
      showNotif('Backend connected ✓');
      await loadDashboardAnalytics();
      await loadCandidateList();
    } catch (e) {
      addLog('INFO', 'Running in standalone mode (backend not reachable)');
    }
  }, 1000);
});
