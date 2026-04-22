import streamlit as st
import joblib
import numpy as np
import pandas as pd

classifier = joblib.load('artifacts/classifier_model.pkl')
regressor  = joblib.load('artifacts/regressor_model.pkl')
le         = joblib.load('artifacts/label_encoder.pkl')

def main():
    st.title('Student Placement Prediction')
    st.info("This app predicts whether a student will be placed in a job and estimates their expected salary (if placed) based on their academic performance, technical skills, and lifestyle factors.")
    
    #Sidebar
    with st.sidebar:
        st.header("Your Academic Information")
        gender = st.selectbox('Gender', ['Male', 'Female'])
        branch = st.selectbox('Branch', ['CSE', 'ECE', 'IT', 'ME', 'CE', 'EEE', 'Other'])
        cgpa   = st.number_input('CGPA', 5.0, 10.0, 7.5, step=0.5)
        tenth_percentage   = st.number_input('10th Percentage', 50.0, 100.0, 75.0, step=0.5)
        twelfth_percentage = st.number_input('12th Percentage', 50.0, 100.0, 75.0, step=0.5)
        backlogs           = st.number_input('Backlogs', 0, 5, 0)
        attendance_percentage = st.number_input('Attendance %', 44.0, 100.0, 75.0, step=0.5)

    #Main Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)  #to create 2 equal-width columns
        with col1:
            st.subheader("Technical Skills")
            coding_skill_rating        = st.number_input('Coding Skill Rating', 1, 5, 3)
            aptitude_skill_rating      = st.number_input('Aptitude Skill Rating', 1, 5, 4)
            communication_skill_rating = st.number_input('Communication Skill Rating', 1, 5, 3)
            projects_completed         = st.number_input('Projects Completed', 0, 8, 5)
            internships_completed      = st.number_input('Internships Completed', 0, 4, 2)
            hackathons_participated    = st.number_input('Hackathons Participated', 0, 6, 3)
            certifications_count       = st.number_input('Certifications Count', 0, 9, 2)

        with col2:
            st.subheader("Lifestyle Factors")
            study_hours_per_day = st.number_input('Study Hours/Day', 0.0, 10.0, 4.0, step=0.5)
            sleep_hours         = st.number_input('Sleep Hours/Day', 4.0, 9.0, 7.0, step=0.5)
            stress_level        = st.number_input('Stress Level (1-10)', 1, 10, 5)
            part_time_job       = st.selectbox('Part-Time Job?', ['No', 'Yes'])
            internet_access     = st.selectbox('Internet Access?', ['Yes', 'No'])
            family_income_level = st.selectbox('Family Income Level', ['High', 'Medium', 'Low'])
            city_tier           = st.selectbox('City Tier', ['Tier 1', 'Tier 2', 'Tier 3'])
            extracurricular_involvement = st.selectbox('Extracurricular Involvement', ['High', 'Medium', 'Low'])

        submitted = st.form_submit_button('Make Prediction')

    if submitted:
        features = {
            "gender": gender,
            "branch": branch,
            "cgpa": cgpa,
            "tenth_percentage": tenth_percentage,
            "twelfth_percentage": twelfth_percentage,
            "backlogs": int(backlogs),
            "study_hours_per_day": study_hours_per_day,
            "attendance_percentage": attendance_percentage,
            "projects_completed": int(projects_completed),
            "internships_completed": int(internships_completed),
            "coding_skill_rating": int(coding_skill_rating),
            "communication_skill_rating": int(communication_skill_rating),
            "aptitude_skill_rating": int(aptitude_skill_rating),
            "hackathons_participated": int(hackathons_participated),
            "certifications_count": int(certifications_count),
            "sleep_hours": sleep_hours,
            "stress_level": int(stress_level),
            "part_time_job": part_time_job,
            "family_income_level": family_income_level,
            "city_tier": city_tier,
            "internet_access": internet_access,
            "extracurricular_involvement": extracurricular_involvement
        }

        result = make_prediction(features)

        placement = result["placement_prediction"]
        salary = result["predicted_salary_lpa"]

        st.subheader("Prediction Results")
        if placement == "Placed":
            st.success(f'Placement Status: {placement}! Congrats!🎉')
            st.markdown("**Keep upskilling and networking. Your career is just getting started!**")
            st.info(f'Estimated Salary: {salary} LPA')
            salary_range = pd.DataFrame({
                "salary_range": {
                    "Min. Salary": 5.18,
                    "Your Salary": salary,
                    "Average Salary": 16.15,
                    "Max Salary": 20.0
                }
            })
            st.bar_chart(salary_range)
        else:
            st.error(f'Placement Status: {placement}')
            st.warning('No salary estimate available for students who were not placed.')
            st.markdown('**Keep improving your skills, your placement is just around the corner! :)**')
            st.subheader("Your Skill Profile")
            skill_data = pd.DataFrame({
                "Your Score": {
                    "CGPA": cgpa,
                    "Coding": coding_skill_rating,
                    "Aptitude": aptitude_skill_rating,
                    "Communication": communication_skill_rating,
                    "Internships": internships_completed,
                    "Projects": projects_completed
                }
            })
            st.area_chart(skill_data)

def make_prediction(features):
    df = pd.DataFrame([features]) #convert dict to df
    placement_label = le.inverse_transform(classifier.predict(df))[0] #get placement prediction & get original labels
    salary = round(float(regressor.predict(df)[0]), 2) if placement_label == 'Placed' else None #predict salary only if Placed
    return {'placement_prediction': placement_label, 'predicted_salary_lpa': salary}


if __name__ == '__main__':
    main()
