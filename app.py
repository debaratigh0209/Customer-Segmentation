#STEP 1: import algorithms 
from flask import Flask, render_template, request, redirect, session, flash, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from sklearn.dummy import DummyClassifier
from werkzeug.utils import secure_filename
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import algorithms

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.static_folder = 'static'
app.secret_key = 'your_secret_key'

class UserProfileForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone_number = StringField('Phone Number', validators=[DataRequired()])

def base_dir():
    return os.path.abspath(os.path.dirname(__file__))

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])

#templates

@app.route('/')
def index():
    return render_template('welcome.html', title="welcome")



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Hardcoded valid username and password
        valid_username = 'admin'
        valid_password = 'admin'
        
        username = request.form['username']
        password = request.form['password']
        if username == valid_username and password == valid_password:
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('login_success'))  # Redirect to the home page
        else:
            flash('Invalid username or password', 'error')
            return render_template('login.html', title="Login")
    else:
        return render_template('login.html', title="Login")

@app.route('/login_success')
def login_success():
    return render_template('login_success.html', title="Login Success")


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        
        # Store user information (you may store it in a database instead)
        session['username'] = username
        session['email'] = email
        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('signup_success'))  # Redirect to the signup success page
    
    return render_template('signup.html', title="Signup", form=form)

@app.route('/signup_success')
def signup_success():
    return render_template('signup_success.html', title="Signup Success")

@app.route('/home')
def home():
    flash("welcome")
    if 'logged_in' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('logout_success'))

@app.route('/logout_success')
def logout_success():
    return render_template('logout.html')



@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(base_dir(), 'static', 'uploads', filename)
            file.save(file_path)
            flash('File uploaded successfully', 'success')
            #return algorithms
            algorithms.algorithms(file_path)

            return render_template('result.html',image=True)
            # Read the content of the CSV file
            df = pd.read_csv(file_path)
            csv_content = df.to_html()
            
            return render_template('segmentation.html', success="File uploaded successfully", 
                                   uploaded_file=filename, csv_content=csv_content)
            
        else:
            flash('Invalid file format. Allowed formats are csv, txt, xls, xlsx', 'error')
            return redirect(request.url)

    return render_template('segmentation.html')



@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html',image=True)


@app.route('/form', methods=['GET'])
def form():
    return render_template('form.html')

@app.route('/user_profile', methods=['GET', 'POST'])
def user_profile():
    form = UserProfileForm()
    if request.method == 'POST' and form.validate_on_submit():
        session['user_data'] = {
            'username': form.username.data,
            'email': form.email.data,
            'phone_number': form.phone_number.data
        }
        return redirect(url_for('user_profile'))
    else:
        user_data = session.get('user_data', {})
        form.username.data = user_data.get('username', '')
        form.email.data = user_data.get('email', '')
        form.phone_number.data = user_data.get('phone_number', '')
        return render_template('user_profile.html', title="User Profile", form=form)

if __name__ == '__main__':
    app.run(debug=True)