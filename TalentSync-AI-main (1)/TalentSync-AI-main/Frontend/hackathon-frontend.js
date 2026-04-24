// hackathon-frontend.js - Connects to your backend
const API_BASE = 'http://localhost:5000';
const API_KEY = localStorage.getItem('TALENTSYNC_API_KEY') || 'demo-key-123456789';

const APP_CONFIG = {
  supabase_url: '',
  supabase_publishable_key: '',
  openai_enabled: false,
  loaded: false
};
let SUPABASE_CLIENT = null;

async function initSupabase() {
  if (!APP_CONFIG.supabase_url || !APP_CONFIG.supabase_publishable_key) {
    console.warn('Supabase config is missing, skipping initialization.');
    return;
  }
  try {
    if (typeof window.supabase === 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js';
      document.head.appendChild(script);
      await new Promise((resolve, reject) => {
        script.onload = resolve;
        script.onerror = reject;
      });
    }
    SUPABASE_CLIENT = window.supabase.createClient(APP_CONFIG.supabase_url, APP_CONFIG.supabase_publishable_key);
    console.log('Supabase client initialized');
  } catch (error) {
    console.warn('Unable to initialize Supabase client:', error);
  }
}

async function loadAppConfig() {
  try {
    const response = await fetch(`${API_BASE}/api/config`, {
      headers: { 'X-API-Key': API_KEY }
    });
    if (!response.ok) throw new Error(`Config load failed (${response.status})`);
    const config = await response.json();
    APP_CONFIG.supabase_url = config.supabase_url || '';
    APP_CONFIG.supabase_publishable_key = config.supabase_publishable_key || '';
    APP_CONFIG.openai_enabled = !!config.openai_enabled;
    APP_CONFIG.loaded = true;
    console.log('App config loaded', APP_CONFIG);
    await initSupabase();
  } catch (error) {
    console.warn('Unable to load app config:', error);
  }
}

loadAppConfig();

// ========== PARSE RESUMES ==========
async function parseResumes(jobDesc, resumeFiles) {
    const formData = new FormData();
    formData.append('job_description', jobDesc);
    
    resumeFiles.forEach(file => formData.append('resumes', file));
    
    try {
        const response = await fetch(`${API_BASE}/api/parse-resumes`, {
            method: 'POST',
            headers: { 'X-API-Key': API_KEY },
            body: formData
        });
        
        const data = await response.json();
        displayResults(data.resumes, data.job_required_skills);
        console.log('✅ Parsed:', data.resumes.length, 'resumes');
    } catch (error) {
        console.error('❌ Parse error:', error);
    }
}

// ========== LOAD SKILL GRAPH ==========
async function loadSkillGraph(category) {
    try {
        const response = await fetch(`${API_BASE}/api/skill-graph/${category}`, {
            headers: { 'X-API-Key': API_KEY }
        });
        
        const graphData = await response.json();
        renderD3Graph(graphData, category);  // Your D3 function
        console.log(`✅ ${category} graph loaded`);
    } catch (error) {
        console.error('❌ Graph error:', error);
    }
}

// ========== CATEGORY BUTTONS ==========
async function loadCategories() {
    try {
        const response = await fetch(`${API_BASE}/api/categories`, {
            headers: { 'X-API-Key': API_KEY }
        });
        const categories = await response.json();
        renderCategoryButtons(categories);
    } catch (error) {
        console.error('❌ Categories error:', error);
    }
}

// ========== UI FUNCTIONS ==========
function displayResults(resumes, jobSkills) {
    const resultsDiv = document.getElementById('results');
    let html = '<h3>📊 Matching Results</h3>';
    
    resumes.forEach((resume, i) => {
        html += `
            <div class="candidate">
                <h4>${resume.name} - ${resume.overall_match_percent}%</h4>
                <p><strong>Role:</strong> ${resume.role} @ ${resume.company}</p>
                <p><strong>Skills:</strong> ${Object.keys(resume.skills_by_category).join(', ')}</p>
            </div>
        `;
    });
    
    html += `<p><strong>Job Skills:</strong> ${jobSkills.join(', ')}</p>`;
    resultsDiv.innerHTML = html;
}

// Hook to your HTML buttons
document.addEventListener('DOMContentLoaded', () => {
    loadCategories();  // Load category buttons
    
    // Parse button
    document.getElementById('parse-btn')?.addEventListener('click', () => {
        const jobDesc = document.getElementById('job-desc').value;
        const files = document.getElementById('resume-files').files;
        parseResumes(jobDesc, Array.from(files));
    });
    
    // Graph buttons (your 4 categories)
    ['Frontend', 'Backend', 'AI/ML', 'DevOps'].forEach(cat => {
        const btn = document.getElementById(`graph-${cat.toLowerCase().replace('/', '-')}`);
        if (btn) btn.addEventListener('click', () => loadSkillGraph(cat));
    });
});