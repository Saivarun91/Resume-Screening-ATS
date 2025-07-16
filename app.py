from flask import Flask, render_template, request, jsonify, make_response
import os
import pdfkit
from werkzeug.utils import secure_filename
from ResumeAnalytics import ResumeAnalytics
import markdown
app = Flask(__name__)
import datetime
import time
import tempfile
from waitress import serve
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'jpeg', 'jpg', 'png', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

analytics = ResumeAnalytics()
shared_data = {}  


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
coverletterdata = ""

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    resume_filename = jd_filename = None
    show_dashboard = False

    ats_score = rrs_score = rfs_score = None
    total_ats_score = breakdown = None
    resume_components = {}
    missing_keywords = missing_skills = None
    resume_tips = course_recommendations = None

    if request.method == 'POST':
        resume = request.files.get('resume')
        jd = request.files.get('jd')
        resume_path = jd_path = None

        if resume and allowed_file(resume.filename):
            resume_filename = secure_filename(resume.filename)
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_filename)
            resume.save(resume_path)

        if jd and allowed_file(jd.filename):
            jd_filename = secure_filename(jd.filename)
            jd_path = os.path.join(app.config['UPLOAD_FOLDER'], jd_filename)
            jd.save(jd_path)
            print(f"[DEBUG] Job description uploaded: {jd_filename}")
        
        rolematch = analytics.getJobRecommendations(resume_path)
        print(f"[DEBUG] Raw rolematch: {rolematch}")

        if isinstance(rolematch, dict):
            if 'ROLEMATCHES' in rolematch:
                shared_data['rolematch'] = rolematch['ROLEMATCHES']
            else:
                shared_data['rolematch'] = rolematch
        else:
            shared_data['rolematch'] = {}

        result = analytics.resumeanalytics(resume_path, jd_path)
        global coverletterdata
        coverletterdata = analytics.getCoverLetter(resume_path, jd_path)

        if jd_path and resume_path:
            cover_letter = coverletterdata
            shared_data['cover_letter'] = cover_letter
            shared_data['cover_letter_preview'] = cover_letter[:150] + "..." if len(cover_letter) > 150 else cover_letter

        if result is None or not isinstance(result, dict):
            return render_template('DASHBOARD.html', error="Error in analysis. Please try again.")

        ats_score = result.get('ATS_SCORE')
        rrs_score = result.get('RESUME_RELEVANCE_SCORE')
        rfs_score = result.get('ROLE_FIT_SCORE')
        resume_tips = result.get('RESUME_TIPS', [])
        course_recommendations = result.get('COURSE_RECOMMENDATIONS', {})

        missing_keywordslist = result.get('MISSING_KEYWORDS', [])
        missing_skillslist = result.get('MISSING_SKILLS', [])
        missing_keywords = len(missing_keywordslist)
        missing_skills = len(missing_skillslist)

        shared_data['missing_keywords'] = missing_keywordslist
        shared_data['missing_skills'] = missing_skillslist
        shared_data['resume_tips'] = resume_tips
        shared_data['course_recommendations'] = course_recommendations

        print("[DEBUG] Course Recommendations:", course_recommendations)

        atsresult = analytics.ATSanalytics(resume_path)

        # ✅ Safely handle atsresult
        if not isinstance(atsresult, dict):
            print("[ERROR] ATSanalytics returned non-dict:", atsresult)
            return render_template('DASHBOARD.html', error="Error in ATS analysis. Please try again.")

        resume_components = atsresult.get('Extracted Data', {})
        ats_score_data = atsresult.get('ATS Score', {})
        total_ats_score = ats_score_data.get('Total Score')
        breakdown = ats_score_data.get('Breakdown', {})
        shared_data['ats_tips'] = atsresult.get('Recommendations', [])

        show_dashboard = True

        return render_template('DASHBOARD.html',
                               resume_filename=resume_filename,
                               jd_filename=jd_filename,
                               show_dashboard=show_dashboard,
                               ats_score=ats_score,
                               total_ats_score=total_ats_score,
                               breakdown=breakdown,
                               rrs_score=rrs_score,
                               rfs_score=rfs_score,
                               resume_components=resume_components,
                               missing_keywords=missing_keywords,
                               missing_skills=missing_skills,
                               resume_tips=resume_tips,
                               course_recommendations=course_recommendations)

    return render_template('DASHBOARD.html', show_dashboard=False)


@app.route('/chatbot', methods=['POST'])
def chatbot_route():
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({'response': 'Please enter a valid message.'})

    try:
        analytics = ResumeAnalytics() 
        response = analytics.chatbot(query=user_query) 
        return jsonify({'response': response})
    except Exception as e:
        print("Error:", e)
        return jsonify({'response': 'Server busy, please try again later.'})


@app.route('/resume-insights')
def show_course_recommendations():
    missing_technical_skills = shared_data.get('missing_keywords', [])
    missing_soft_skills = shared_data.get('missing_skills', [])
    resume_tips = shared_data.get('resume_tips', [])
    courseRecommendations = shared_data.get('course_recommendations', {})

    cover_letter_preview = shared_data.get(
        'cover_letter_preview',
        'No cover letter available. Please upload resume and job description to generate one.'
    )
    print("[DEBUG] Course Recommendations:", courseRecommendations)
    print("[DEBUG] Cover Letter Preview:", cover_letter_preview)

    
    resume_tips = shared_data.get('resume_tips', []) + shared_data.get('ats_tips', [])
    import random
    random.shuffle(resume_tips)
    resume_tips_html = [markdown.markdown(tip) for tip in resume_tips]
    role_matches = shared_data.get('rolematch', {})
    print("[DEBUG] Role Matches:", role_matches)
    # Render the page
    return render_template(
        'course_recommendations.html',
        missing_technical_skills=missing_technical_skills,
        missing_soft_skills=missing_soft_skills,
        role_matches=role_matches,
        resume_tips=resume_tips_html,
        cover_letter_preview=cover_letter_preview
    )


@app.route('/course-recommendations')
def courseRecommendations():
    raw_course_data = shared_data.get('course_recommendations', {})
    
    paid_course_recommendations = {}
    youtube_course_recommendations = {}

    # Separate YouTube vs paid platforms
    for topic, courses in raw_course_data.items():
        paid_courses = []
        youtube_courses = []

        for course in courses:
            if course['PLATFORM'].lower() == 'youtube':
                youtube_courses.append(course)
            else:
                paid_courses.append(course)

        if paid_courses:
            paid_course_recommendations[topic] = paid_courses
        if youtube_courses:
            youtube_course_recommendations[topic] = youtube_courses 

    return render_template(
        'recommendations.html',  # Make sure this matches your template file name
        paid_course_recommendations=paid_course_recommendations,
        youtube_course_recommendations=youtube_course_recommendations
    )

# === Helper Function ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Routes ===
current_documents = []  
@app.route('/PDFChatbot', methods=['GET', 'POST'])
def PDFChatbot():
    global current_documents
    
    if request.method == 'POST':
       
        uploaded_files = request.files.getlist('documents')
        query = request.form.get('query', '').strip()
        
        new_files = []
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                if filepath not in current_documents:  # Avoid duplicates
                    new_files.append(filepath)
        
        current_documents.extend(new_files)  # Add new files to global list
        
        if not query and current_documents:
            query = "Please analyze my resume."
        elif not query:
            return jsonify({"response": "Please upload a resume or ask a question."})
        
        try:
            response = analytics.pdfchatbot(current_documents, query)
            return jsonify({"response": response})
        except Exception as e:
            app.logger.error(f"Error in PDFChatbot: {str(e)}")
            return jsonify({"response": "Error processing request. Please try again."}), 500
    
    return render_template('chatbot.html')

@app.route('/remove-file', methods=['POST'])
def remove_file():
    global current_documents
    data = request.get_json()
    filename = data.get('filename')
    
    if filename in current_documents:
        current_documents.remove(filename)
        try:
            os.remove(filename)
        except OSError:
            pass
    
    return jsonify({"status": "success"})

from flask import Flask, render_template, request, make_response
import pdfkit
from werkzeug.utils import secure_filename
import json

@app.route('/api/download-cover-letter', methods=['GET'])  # Match the frontend URL and method
def download_cover_letter():
    try:
        global coverletterdata
        cover_letter_text = coverletterdata
        response = make_response(cover_letter_text)
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Content-Disposition'] = 'attachment; filename=CoverLetter.txt'
        return response
    except Exception as e:
        return str(e), 500

@app.route('/customcoverletter', methods=['GET', 'POST'])
def custom_cover_letter():
    if request.method == 'POST':
        # Check if this is a PDF generation request
        if 'html' in request.form:
            html_content = request.form.get('html')
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as output:
                pdfkit.from_string(html_content, output.name)
                output.seek(0)
                
                response = make_response(output.read())
                response.headers['Content-Type'] = 'application/pdf'
                response.headers['Content-Disposition'] = 'attachment; filename=cover_letter.pdf'
                
                # Clean up
                os.unlink(output.name)
                
                return response
        
        # Normal cover letter generation
        job_title = request.form.get('jobTitle')
        company_name = request.form.get('companyName')
        your_name = request.form.get('yourName')
        additional_info = request.form.get('coverLetterContent')

        cover_letter_content = analytics.getCustomCoverLetter(
            job_title, company_name, your_name, additional_info
        )

        if cover_letter_content.startswith('{'):
            try:
                data = json.loads(cover_letter_content)
                if 'cover_letter' in data:
                    cover_letter_content = data['cover_letter']
            except json.JSONDecodeError:
                pass

        return render_template("Cover letter.html", 
                            generated=True,
                            letter_content=cover_letter_content)
    
    return render_template("Cover letter.html", generated=False)
    

@app.route('/')
def landing_page():
    return render_template('Landing Page.html')

if __name__ == '__main__':
   
    serve(app, host='0.0.0.0', port=10000)