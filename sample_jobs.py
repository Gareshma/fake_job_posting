"""
Sample job descriptions - 10 real and 10 fake job postings
"""

SAMPLE_JOBS = {
    0:  {
        "title": "Marketing Manager",
        "description": "Join our marketing team as a Marketing Manager where you'll develop and execute marketing strategies to drive brand awareness and customer acquisition. You'll manage social media campaigns, analyze market trends, and collaborate with cross-functional teams. We're looking for someone with 5+ years of marketing experience and strong analytical skills.",
        "company": "GrowthCo Inc",
        "requirements": "Marketing degree, 5+ years experience, Google Analytics certified",
        "label": "real"
    },
    
    1: {
        "title": "Easy Money Data Entry - Work From Home!!!",
        "description": "MAKE $5000/WEEK FROM HOME!!! No experience needed! Just copy and paste data for 2 hours daily. Immediate start! Send $99 processing fee to secure your position. Guaranteed income! Limited spots available! Contact immediately with credit card info ready!!!",
        "company": "QuickCash Solutions",
        "requirements": "No experience needed! Just $99 startup fee!",
        "label": "fake"
    },

    2: {
    "title": "Data Analyst - Business Intelligence",
    "description": "A leading retail analytics firm is looking for a Data Analyst to support business intelligence reporting and insights. Responsibilities include building dashboards using Tableau, analyzing customer data, and presenting actionable insights to stakeholders. Candidates should have 2+ years of analytics experience and a strong grasp of SQL and Excel. Experience with Python and cloud tools is a plus. We offer healthcare, 401(k), remote flexibility, and opportunities for growth.",
    "company": "RetailIQ Analytics",
    "requirements": "Bachelor's in Statistics or Data Science, 2+ years in analytics, SQL, Tableau, Excel",
    "label": "Fake"
},
    
    3: {
        "title": "Senior Accountant",
        "description": "Established accounting firm seeks experienced Senior Accountant for full-time position. Responsibilities include preparing financial statements, managing accounts payable/receivable, and ensuring compliance with regulations. CPA certification preferred. Excellent benefits package including 401k matching and professional development opportunities.",
        "company": "Miller & Associates CPA",
        "requirements": "CPA preferred, 4+ years accounting experience",
        "label": "real"
    },
    
    4: {
        "title": "Product Tester - Get Paid to Shop!",
        "description": "Mystery shopper needed! Shop at major retailers and keep everything FREE! Plus get paid $200 per assignment! Easy work! Just need to cash checks and wire remaining funds. No background check needed! Start immediately!",
        "company": "Consumer Testing Group",
        "requirements": "Must be able to cash checks and wire money",
        "label": "fake"
    },
    
    5: {
        "title": "Project Manager",
        "description": "Seeking an experienced Project Manager to lead cross-functional teams and deliver projects on time and within budget. You'll use Agile methodologies, manage stakeholder relationships, and ensure project quality. PMP certification is a plus. This role offers growth opportunities in a dynamic environment.",
        "company": "Enterprise Solutions Ltd",
        "requirements": "PMP certification preferred, 5+ years project management",
        "label": "real"
    },
    
    6: {
        "title": "Sales Representative",
        "description": "Dynamic sales opportunity for motivated individual to join our B2B sales team. You'll identify prospects, build relationships, and close deals in the technology sector. Base salary plus commission structure with unlimited earning potential. Training provided for the right candidate.",
        "company": "TechSales Pro",
        "requirements": "Sales experience preferred, strong communication skills",
        "label": "real"
    },
    
    7: {
        "title": "Registered Nurse",
        "description": "Hospital seeking compassionate Registered Nurse for medical-surgical unit. Provide direct patient care, administer medications, and collaborate with healthcare team. Current RN license required. We offer competitive wages, comprehensive benefits, and opportunities for continuing education.",
        "company": "Metro General Hospital",
        "requirements": "Current RN license, BLS certification",
        "label": "real"
    },
    
    8: {
        "title": "Operations Manager",
        "description": "Operations Manager position available for manufacturing facility. Oversee daily operations, manage production schedules, and ensure quality standards. Lean Six Sigma experience preferred. Leadership experience in manufacturing environment required. Excellent benefits and growth opportunities.",
        "company": "Manufacturing Excellence Corp",
        "requirements": "Operations experience, Lean Six Sigma preferred",
        "label": "real"
    },
    
    9: {
        "title": "Elementary School Teacher",
        "description": "Elementary school seeking dedicated teacher for 3rd grade classroom. Create engaging lesson plans, assess student progress, and maintain positive classroom environment. State teaching certification required. We support professional development and offer summers off with competitive teacher benefits.",
        "company": "Riverside Elementary School",
        "requirements": "Teaching certification, Elementary education degree",
        "label": "real"
    },
    
    10: {
        "title": "Data Analyst",
        "description": "We're hiring a Data Analyst to help us make data-driven decisions. You'll work with large datasets, create visualizations using Tableau and Python, and present insights to stakeholders. The ideal candidate has experience with SQL, Python, and statistical analysis. Remote work options available.",
        "company": "DataVision Analytics",
        "requirements": "Statistics or related degree, SQL proficiency, Python experience",
        "label": "real"
    },
    
    11: {
        "title": "Personal Assistant to CEO - $8000/month",
        "description": "Urgent! CEO needs personal assistant ASAP. Work from home, flexible hours. Handle emails and schedule meetings. Must be trustworthy for financial transactions. Will provide company credit card for expenses. Send personal information and bank details for payroll setup.",
        "company": "Global Enterprises International",
        "requirements": "Must provide bank account information",
        "label": "fake"
    },
    
    12: {
        "title": "UX/UI Designer",
        "description": "Creative UX/UI Designer needed to design intuitive user interfaces for our mobile and web applications. You'll conduct user research, create wireframes and prototypes, and collaborate with developers. Portfolio showcasing mobile-first design is required. We value creativity, attention to detail, and user-centered design principles.",
        "company": "DesignHub Creative",
        "requirements": "Design degree, 3+ years UX/UI experience, portfolio required",
        "label": "real"
    },
    
    13: {
        "title": "Investment Opportunity Representative",
        "description": "Join our exclusive investment program! Earn $10,000+ monthly by recruiting others! No selling required! Just invite friends and family to invest minimum $500. Guaranteed returns! Multi-level structure ensures unlimited income potential! Act fast - limited positions!",
        "company": "Wealth Builders Network",
        "requirements": "Initial investment of $500 required",
        "label": "fake"
    },
    
    14: {
        "title": "Remote Customer Service - Immediate Start",
        "description": "Customer service position available immediately! $25/hour to start! Work from home! Handle customer inquiries via email. Must purchase company equipment package for $150. Training materials included. Send payment via wire transfer or gift cards only.",
        "company": "Customer Care Solutions LLC",
        "requirements": "$150 equipment fee required upfront",
        "label": "fake"
    },
    
    15: {
        "title": "Government Grant Processor",
        "description": "Process government grants from home! $4000/week guaranteed! Help people get free government money! No experience necessary! Just forward applications and collect processing fees. Send $75 registration fee to get started immediately!",
        "company": "Federal Grant Processing",
        "requirements": "$75 registration fee, government clearance",
        "label": "fake"
    },
    
    16: {
        "title": "Social Media Influencer Manager - Urgent",
        "description": "Manage Instagram accounts for celebrities! $6000/month! Must have 10,000+ followers and provide login credentials for verification. Handle their DMs and post content. Wire transfer payment weekly. Send social media passwords for background check.",
        "company": "Celebrity Social Management",
        "requirements": "Must provide social media login credentials",
        "label": "fake"
    },
    
    17: {
        "title": "Package Redistribution Specialist",
        "description": "Receive packages at your home and reship them! Easy $300 per package! Help international customers who can't ship directly to their country. No questions asked about package contents. Just forward packages and keep the cash!",
        "company": "International Shipping Solutions",
        "requirements": "Must accept packages of unknown contents",
        "label": "fake"
    },
    
    18: {
        "title": "Cryptocurrency Investment Advisor",
        "description": "New crypto opportunity! Make $50,000 in 30 days guaranteed! Our AI trading system never loses! Just deposit minimum $1000 and watch it grow! Refer friends for bonus payments! Limited time offer! Send Bitcoin payment to secure your spot!",
        "company": "Crypto Wealth Systems",
        "requirements": "Minimum $1000 cryptocurrency deposit",
        "label": "fake"
    },
    
    19: {
        "title": "Medical Claims Processor - No Experience Needed",
        "description": "Process medical claims from home! $40/hour! No medical background required! Just forward patient information to processing center. Must purchase HIPAA compliance software for $200. Handle sensitive medical records. Training via unsecured email.",
        "company": "MedClaims Processing Inc",
        "requirements": "$200 software fee, handle confidential medical data",
        "label": "fake"
    },
}

def get_job_list():
    """Return a list of jobs for dropdown display"""
    return [
        {
            'id': job_id,
            'display_text': f"{job_data['title']} - {job_data['company']}",
            'label': job_data['label']
        }
        for job_id, job_data in SAMPLE_JOBS.items()
    ]

def get_job_by_id(job_id):
    """Get job details by ID"""
    return SAMPLE_JOBS.get(job_id)