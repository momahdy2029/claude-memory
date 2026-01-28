"""
Agent Registry - Defines all available sub-agents with metadata.
"""

AGENT_CATEGORIES = {
    "engineering": {
        "name": "Engineering",
        "icon": "code",
        "color": "#58a6ff",
        "description": "Software development and architecture agents"
    },
    "testing": {
        "name": "Testing & QA",
        "icon": "shield-check",
        "color": "#3fb950",
        "description": "Quality assurance and testing specialists"
    },
    "design": {
        "name": "Design & UX",
        "icon": "palette",
        "color": "#a371f7",
        "description": "UI/UX design and visual specialists"
    },
    "product": {
        "name": "Product",
        "icon": "lightbulb",
        "color": "#d29922",
        "description": "Product management and strategy"
    },
    "marketing": {
        "name": "Marketing",
        "icon": "megaphone",
        "color": "#f85149",
        "description": "Marketing and growth specialists"
    },
    "operations": {
        "name": "Operations",
        "icon": "settings",
        "color": "#8b949e",
        "description": "DevOps and operations management"
    },
    "data": {
        "name": "Data & Analytics",
        "icon": "chart-bar",
        "color": "#39d353",
        "description": "Data analysis and reporting"
    },
    "xr": {
        "name": "XR & Spatial",
        "icon": "cube",
        "color": "#bf4b8a",
        "description": "AR/VR/XR and spatial computing"
    },
    "exploration": {
        "name": "Exploration",
        "icon": "search",
        "color": "#79c0ff",
        "description": "Codebase exploration and research"
    }
}

AVAILABLE_AGENTS = [
    # Exploration
    {
        "id": "Explore",
        "name": "Explore",
        "category": "exploration",
        "description": "Fast codebase exploration specialist. Finds files, searches code, answers questions about architecture.",
        "tags": ["search", "codebase", "quick"],
        "default_enabled": True,
        "priority": 10
    },
    {
        "id": "Plan",
        "name": "Plan",
        "category": "exploration",
        "description": "Software architect for designing implementation plans. Creates step-by-step strategies.",
        "tags": ["planning", "architecture", "strategy"],
        "default_enabled": True,
        "priority": 9
    },

    # Engineering - Core
    {
        "id": "general-purpose",
        "name": "General Purpose",
        "category": "engineering",
        "description": "Multi-step task handler for complex research and code execution.",
        "tags": ["general", "multi-step", "research"],
        "default_enabled": True,
        "priority": 10
    },
    {
        "id": "Bash",
        "name": "Bash Specialist",
        "category": "engineering",
        "description": "Command execution specialist for git, terminal operations, and system tasks.",
        "tags": ["bash", "git", "terminal"],
        "default_enabled": True,
        "priority": 9
    },
    {
        "id": "engineering-senior-developer",
        "name": "Senior Developer",
        "category": "engineering",
        "description": "Premium implementation specialist. Masters Laravel/Livewire/FluxUI, advanced CSS, Three.js.",
        "tags": ["laravel", "livewire", "css", "threejs"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "engineering-ai-engineer",
        "name": "AI/ML Engineer",
        "category": "engineering",
        "description": "Machine learning model development, deployment, and integration into production systems.",
        "tags": ["ml", "ai", "models", "data-pipelines"],
        "default_enabled": True,
        "priority": 7
    },
    {
        "id": "Backend Architect",
        "name": "Backend Architect",
        "category": "engineering",
        "description": "Scalable system design, database architecture, API development, cloud infrastructure.",
        "tags": ["backend", "api", "database", "cloud"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "Frontend Developer",
        "name": "Frontend Developer",
        "category": "engineering",
        "description": "Modern web technologies, React/Vue/Angular frameworks, UI implementation.",
        "tags": ["frontend", "react", "vue", "angular", "ui"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "kotlin-material3-dev",
        "name": "Kotlin Material3 Dev",
        "category": "engineering",
        "description": "Kotlin development with Material3 UI, color schemes, theming, and visual design.",
        "tags": ["kotlin", "android", "material3", "mobile"],
        "default_enabled": False,
        "priority": 6
    },
    {
        "id": "Mobile App Builder",
        "name": "Mobile App Builder",
        "category": "engineering",
        "description": "Native iOS/Android development and cross-platform frameworks.",
        "tags": ["mobile", "ios", "android", "react-native", "flutter"],
        "default_enabled": False,
        "priority": 6
    },
    {
        "id": "Rapid Prototyper",
        "name": "Rapid Prototyper",
        "category": "engineering",
        "description": "Ultra-fast proof-of-concept and MVP creation.",
        "tags": ["prototype", "mvp", "fast", "poc"],
        "default_enabled": True,
        "priority": 7
    },
    {
        "id": "LSP/Index Engineer",
        "name": "LSP/Index Engineer",
        "category": "engineering",
        "description": "Language Server Protocol specialist for code intelligence systems.",
        "tags": ["lsp", "indexing", "code-intelligence"],
        "default_enabled": False,
        "priority": 5
    },

    # Feature Development
    {
        "id": "feature-dev:code-architect",
        "name": "Code Architect",
        "category": "engineering",
        "description": "Designs feature architectures by analyzing codebase patterns and conventions.",
        "tags": ["architecture", "patterns", "design"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "feature-dev:code-explorer",
        "name": "Code Explorer",
        "category": "engineering",
        "description": "Deep analysis of existing features, execution paths, and dependencies.",
        "tags": ["analysis", "tracing", "dependencies"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "feature-dev:code-reviewer",
        "name": "Code Reviewer",
        "category": "engineering",
        "description": "Reviews code for bugs, security vulnerabilities, and quality issues.",
        "tags": ["review", "security", "quality"],
        "default_enabled": True,
        "priority": 8
    },

    # Testing & QA
    {
        "id": "testing-reality-checker",
        "name": "Reality Checker",
        "category": "testing",
        "description": "Evidence-based certification. Requires overwhelming proof for production readiness.",
        "tags": ["testing", "verification", "strict"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "EvidenceQA",
        "name": "Evidence QA",
        "category": "testing",
        "description": "Screenshot-obsessed QA specialist. Requires visual proof for everything.",
        "tags": ["qa", "screenshots", "visual-testing"],
        "default_enabled": True,
        "priority": 7
    },
    {
        "id": "API Tester",
        "name": "API Tester",
        "category": "testing",
        "description": "Comprehensive API validation, performance testing, and quality assurance.",
        "tags": ["api", "testing", "performance"],
        "default_enabled": True,
        "priority": 7
    },
    {
        "id": "Test Results Analyzer",
        "name": "Test Results Analyzer",
        "category": "testing",
        "description": "Test result evaluation, quality metrics analysis, and actionable insights.",
        "tags": ["analysis", "metrics", "insights"],
        "default_enabled": True,
        "priority": 6
    },
    {
        "id": "Performance Benchmarker",
        "name": "Performance Benchmarker",
        "category": "testing",
        "description": "Performance testing and optimization across all applications.",
        "tags": ["performance", "benchmarking", "optimization"],
        "default_enabled": True,
        "priority": 7
    },

    # Operations
    {
        "id": "DevOps Automator",
        "name": "DevOps Automator",
        "category": "operations",
        "description": "Infrastructure automation, CI/CD pipeline development, and cloud operations.",
        "tags": ["devops", "ci-cd", "automation", "cloud"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "Infrastructure Maintainer",
        "name": "Infrastructure Maintainer",
        "category": "operations",
        "description": "System reliability, performance optimization, and technical operations.",
        "tags": ["infrastructure", "reliability", "maintenance"],
        "default_enabled": True,
        "priority": 7
    },
    {
        "id": "Workflow Optimizer",
        "name": "Workflow Optimizer",
        "category": "operations",
        "description": "Process improvement specialist for maximum productivity and efficiency.",
        "tags": ["workflow", "optimization", "automation"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "Tool Evaluator",
        "name": "Tool Evaluator",
        "category": "operations",
        "description": "Technology assessment specialist for tools, software, and platforms.",
        "tags": ["evaluation", "tools", "assessment"],
        "default_enabled": False,
        "priority": 5
    },

    # Design & UX
    {
        "id": "ArchitectUX",
        "name": "Architect UX",
        "category": "design",
        "description": "Technical architecture and UX specialist with CSS systems expertise.",
        "tags": ["ux", "architecture", "css"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "UI Designer",
        "name": "UI Designer",
        "category": "design",
        "description": "Visual design systems, component libraries, and pixel-perfect interfaces.",
        "tags": ["ui", "design-systems", "components"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "UX Researcher",
        "name": "UX Researcher",
        "category": "design",
        "description": "User behavior analysis, usability testing, and data-driven design insights.",
        "tags": ["research", "usability", "user-behavior"],
        "default_enabled": True,
        "priority": 7
    },
    {
        "id": "Brand Guardian",
        "name": "Brand Guardian",
        "category": "design",
        "description": "Brand identity development, consistency maintenance, and strategic positioning.",
        "tags": ["brand", "identity", "consistency"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "design-visual-storyteller",
        "name": "Visual Storyteller",
        "category": "design",
        "description": "Visual narratives, multimedia content, and brand storytelling through design.",
        "tags": ["storytelling", "multimedia", "visual"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "Whimsy Injector",
        "name": "Whimsy Injector",
        "category": "design",
        "description": "Adds personality, delight, and playful elements to brand experiences.",
        "tags": ["whimsy", "delight", "personality"],
        "default_enabled": False,
        "priority": 4
    },

    # Product
    {
        "id": "project-manager-senior",
        "name": "Senior Project Manager",
        "category": "product",
        "description": "Converts specs to tasks with realistic scope and exact requirements.",
        "tags": ["project-management", "specs", "tasks"],
        "default_enabled": True,
        "priority": 8
    },
    {
        "id": "product-sprint-prioritizer",
        "name": "Sprint Prioritizer",
        "category": "product",
        "description": "Agile sprint planning, feature prioritization, and resource allocation.",
        "tags": ["agile", "sprint", "prioritization"],
        "default_enabled": True,
        "priority": 7
    },
    {
        "id": "product-feedback-synthesizer",
        "name": "Feedback Synthesizer",
        "category": "product",
        "description": "Collects and analyzes user feedback to extract actionable product insights.",
        "tags": ["feedback", "analysis", "insights"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "product-trend-researcher",
        "name": "Trend Researcher",
        "category": "product",
        "description": "Market intelligence, emerging trends, and competitive analysis.",
        "tags": ["trends", "market", "competitive"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "Studio Operations",
        "name": "Studio Operations",
        "category": "product",
        "description": "Day-to-day studio efficiency, process optimization, and resource coordination.",
        "tags": ["operations", "efficiency", "coordination"],
        "default_enabled": False,
        "priority": 4
    },
    {
        "id": "Project Shepherd",
        "name": "Project Shepherd",
        "category": "product",
        "description": "Cross-functional project coordination and stakeholder alignment.",
        "tags": ["coordination", "cross-functional", "stakeholders"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "Experiment Tracker",
        "name": "Experiment Tracker",
        "category": "product",
        "description": "Experiment design, A/B testing, and hypothesis validation.",
        "tags": ["experiments", "ab-testing", "validation"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "Studio Producer",
        "name": "Studio Producer",
        "category": "product",
        "description": "High-level creative and technical project orchestration.",
        "tags": ["production", "creative", "orchestration"],
        "default_enabled": False,
        "priority": 5
    },

    # Marketing
    {
        "id": "marketing-growth-hacker",
        "name": "Growth Hacker",
        "category": "marketing",
        "description": "Rapid user acquisition through data-driven experimentation and viral loops.",
        "tags": ["growth", "acquisition", "viral"],
        "default_enabled": False,
        "priority": 6
    },
    {
        "id": "marketing-content-creator",
        "name": "Content Creator",
        "category": "marketing",
        "description": "Multi-platform campaigns, editorial calendars, and brand storytelling.",
        "tags": ["content", "campaigns", "storytelling"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "marketing-social-media-strategist",
        "name": "Social Media Strategist",
        "category": "marketing",
        "description": "Twitter, LinkedIn strategies with viral campaigns and thought leadership.",
        "tags": ["social-media", "linkedin", "twitter"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "marketing-instagram-curator",
        "name": "Instagram Curator",
        "category": "marketing",
        "description": "Visual storytelling, community building, and multi-format content for Instagram.",
        "tags": ["instagram", "visual", "community"],
        "default_enabled": False,
        "priority": 4
    },
    {
        "id": "marketing-tiktok-strategist",
        "name": "TikTok Strategist",
        "category": "marketing",
        "description": "Viral content creation, algorithm optimization, and TikTok community building.",
        "tags": ["tiktok", "viral", "algorithm"],
        "default_enabled": False,
        "priority": 4
    },
    {
        "id": "marketing-twitter-engager",
        "name": "Twitter Engager",
        "category": "marketing",
        "description": "Real-time engagement, thought leadership, and community-driven growth.",
        "tags": ["twitter", "engagement", "thought-leadership"],
        "default_enabled": False,
        "priority": 4
    },
    {
        "id": "marketing-reddit-community-builder",
        "name": "Reddit Community Builder",
        "category": "marketing",
        "description": "Authentic Reddit engagement, value-driven content, and community building.",
        "tags": ["reddit", "community", "authentic"],
        "default_enabled": False,
        "priority": 4
    },
    {
        "id": "App Store Optimizer",
        "name": "App Store Optimizer",
        "category": "marketing",
        "description": "ASO, conversion rate optimization, and app discoverability.",
        "tags": ["aso", "app-store", "conversion"],
        "default_enabled": False,
        "priority": 4
    },

    # Data & Analytics
    {
        "id": "data-analytics-reporter",
        "name": "Analytics Reporter",
        "category": "data",
        "description": "Transforms raw data into actionable business insights and dashboards.",
        "tags": ["analytics", "dashboards", "insights"],
        "default_enabled": True,
        "priority": 7
    },
    {
        "id": "Analytics Reporter",
        "name": "Analytics Reporter",
        "category": "data",
        "description": "Data analysis, KPI tracking, and strategic decision support.",
        "tags": ["analytics", "kpi", "reporting"],
        "default_enabled": True,
        "priority": 7
    },
    {
        "id": "Finance Tracker",
        "name": "Finance Tracker",
        "category": "data",
        "description": "Financial planning, budget management, and performance analysis.",
        "tags": ["finance", "budget", "planning"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "Executive Summary Generator",
        "name": "Executive Summary Generator",
        "category": "data",
        "description": "McKinsey-style executive summaries with SCQA and Pyramid Principle frameworks.",
        "tags": ["executive", "summary", "frameworks"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "Legal Compliance Checker",
        "name": "Legal Compliance Checker",
        "category": "data",
        "description": "Ensures compliance with laws, regulations, and industry standards.",
        "tags": ["legal", "compliance", "regulations"],
        "default_enabled": False,
        "priority": 4
    },
    {
        "id": "Support Responder",
        "name": "Support Responder",
        "category": "data",
        "description": "Customer support, issue resolution, and multi-channel service.",
        "tags": ["support", "customer-service", "issues"],
        "default_enabled": False,
        "priority": 4
    },

    # XR & Spatial
    {
        "id": "XR Interface Architect",
        "name": "XR Interface Architect",
        "category": "xr",
        "description": "Spatial interaction design for immersive AR/VR/XR environments.",
        "tags": ["xr", "spatial", "interaction"],
        "default_enabled": False,
        "priority": 6
    },
    {
        "id": "XR Immersive Developer",
        "name": "XR Immersive Developer",
        "category": "xr",
        "description": "WebXR and browser-based AR/VR/XR applications.",
        "tags": ["webxr", "browser", "immersive"],
        "default_enabled": False,
        "priority": 6
    },
    {
        "id": "XR Cockpit Interaction Specialist",
        "name": "XR Cockpit Specialist",
        "category": "xr",
        "description": "Cockpit-based control systems for XR environments.",
        "tags": ["cockpit", "controls", "xr"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "visionos-spatial-engineer",
        "name": "VisionOS Spatial Engineer",
        "category": "xr",
        "description": "Native visionOS spatial computing with SwiftUI and Liquid Glass design.",
        "tags": ["visionos", "swiftui", "spatial"],
        "default_enabled": False,
        "priority": 6
    },
    {
        "id": "terminal-integration-specialist",
        "name": "Terminal Integration Specialist",
        "category": "xr",
        "description": "Terminal emulation, SwiftTerm integration, and VT100/xterm standards.",
        "tags": ["terminal", "swiftterm", "integration"],
        "default_enabled": False,
        "priority": 5
    },
    {
        "id": "macOS Spatial/Metal Engineer",
        "name": "macOS Metal Engineer",
        "category": "xr",
        "description": "Native Swift and Metal for high-performance 3D rendering on macOS and Vision Pro.",
        "tags": ["metal", "swift", "3d", "macos"],
        "default_enabled": False,
        "priority": 6
    },

    # Special
    {
        "id": "agents-orchestrator",
        "name": "Agents Orchestrator",
        "category": "operations",
        "description": "Autonomous pipeline manager that orchestrates the entire development workflow.",
        "tags": ["orchestration", "pipeline", "automation"],
        "default_enabled": True,
        "priority": 9
    },
    {
        "id": "claude-code-guide",
        "name": "Claude Code Guide",
        "category": "exploration",
        "description": "Expert on Claude Code features, hooks, MCP servers, settings, and integrations.",
        "tags": ["claude-code", "help", "guide"],
        "default_enabled": True,
        "priority": 8
    }
]

# MCP Servers
AVAILABLE_MCPS = [
    {
        "id": "claude-memory",
        "name": "Claude Memory",
        "description": "Semantic memory storage and retrieval for persistent context.",
        "icon": "brain",
        "color": "#a371f7",
        "default_enabled": True
    },
    {
        "id": "context7",
        "name": "Context7",
        "description": "Up-to-date documentation and API references from the web.",
        "icon": "book",
        "color": "#58a6ff",
        "default_enabled": True
    },
    {
        "id": "filesystem",
        "name": "Filesystem",
        "description": "Enhanced file system operations and management.",
        "icon": "folder",
        "color": "#d29922",
        "default_enabled": False
    },
    {
        "id": "github",
        "name": "GitHub",
        "description": "GitHub API integration for repos, issues, and PRs.",
        "icon": "github",
        "color": "#8b949e",
        "default_enabled": False
    },
    {
        "id": "postgres",
        "name": "PostgreSQL",
        "description": "Direct PostgreSQL database access and queries.",
        "icon": "database",
        "color": "#336791",
        "default_enabled": False
    },
    {
        "id": "puppeteer",
        "name": "Puppeteer",
        "description": "Browser automation and web scraping capabilities.",
        "icon": "globe",
        "color": "#3fb950",
        "default_enabled": False
    }
]

# Hooks
AVAILABLE_HOOKS = [
    {
        "id": "grounding-hook",
        "name": "Grounding Hook",
        "description": "Injects context and anchors before every response to prevent hallucinations.",
        "trigger": "UserPromptSubmit",
        "icon": "anchor",
        "color": "#a371f7",
        "default_enabled": True
    },
    {
        "id": "pre-tool-check",
        "name": "Pre-Tool Check",
        "description": "Blocks actions that violate anchors or entity registry BEFORE execution.",
        "trigger": "PreToolUse",
        "icon": "shield",
        "color": "#d29922",
        "default_enabled": True
    },
    {
        "id": "detect-correction",
        "name": "Detect Correction",
        "description": "Auto-detects when user corrects Claude and creates anchors.",
        "trigger": "UserPromptSubmit",
        "icon": "check-circle",
        "color": "#3fb950",
        "default_enabled": True
    },
    {
        "id": "log-tool-use",
        "name": "Log Tool Use",
        "description": "Logs all tool usage to the timeline for tracking.",
        "trigger": "PostToolUse",
        "icon": "file-text",
        "color": "#58a6ff",
        "default_enabled": True
    },
    {
        "id": "auto-detect-response",
        "name": "Auto-Detect Response",
        "description": "Extracts decisions and observations from Claude's responses using LLM.",
        "trigger": "Stop",
        "icon": "brain",
        "color": "#f85149",
        "default_enabled": True
    },
    {
        "id": "log-user-request",
        "name": "Log User Request",
        "description": "Logs user requests to the timeline.",
        "trigger": "UserPromptSubmit",
        "icon": "message-circle",
        "color": "#8b949e",
        "default_enabled": True
    }
]


def get_agents_by_category():
    """Group agents by category."""
    result = {}
    for agent in AVAILABLE_AGENTS:
        cat = agent["category"]
        if cat not in result:
            result[cat] = []
        result[cat].append(agent)
    return result


def get_agent_by_id(agent_id: str):
    """Get agent by ID."""
    for agent in AVAILABLE_AGENTS:
        if agent["id"] == agent_id:
            return agent
    return None
