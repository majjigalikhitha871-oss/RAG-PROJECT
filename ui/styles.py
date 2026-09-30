APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink: #17251d; --muted: #65736a; --leaf: #286847; --leaf-dark: #1c5036; --lime: #d2ed88; --paper: #f4f6f0; --line: #dce3d9; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: radial-gradient(ellipse at 85% 0%, rgba(210,237,136,.22), transparent 31%), var(--paper); }
h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: 0; }
h1 { font-size: 2.35rem !important; font-weight: 800 !important; }
[data-testid="stSidebar"] { background: #e9eee5; border-right: 1px solid var(--line); }
[data-testid="stMetric"] { background: rgba(255,255,255,.72); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
.status-line { color: var(--muted); font-size: .9rem; }
.brand-rule { height: 4px; width: 54px; background: var(--leaf); border-radius: 4px; margin: 0 0 1rem; }
.stButton > button[kind="primary"] { background: var(--leaf); border-color: var(--leaf); }
.stButton > button[kind="primary"]:hover { background: var(--leaf-dark); border-color: var(--leaf-dark); }
div[data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.7); }
div[data-testid="stExpander"] { border-color: var(--line); border-radius: 8px; }
</style>
"""