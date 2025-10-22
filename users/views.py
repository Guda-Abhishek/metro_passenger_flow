from django.shortcuts import render, redirect
from .models import RegisteredUser
from django.core.files.storage import FileSystemStorage

def register_view(request):
    msg = ''
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        mobile = request.POST.get('mobile')
        password = request.POST.get('password')
        image = request.FILES.get('image')

        # Basic validation
        if not (name and email and mobile and password and image):
            msg = "All fields are required."
        else:
            # Save image manually
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            img_url = fs.url(filename)

            # Save user with is_active=False
            RegisteredUser.objects.create(
                name=name,
                email=email,
                mobile=mobile,
                password=password,
                image=filename,
                is_active=False
            )
            msg = "Registered successfully! Wait for admin approval."

    return render(request, 'register.html', {'msg': msg})

from django.utils import timezone

from django.utils import timezone
import pytz

def user_login(request):
    msg = ''
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        try:
            user = RegisteredUser.objects.get(name=name, password=password)
            if user.is_active:
                # Convert current time to IST
                ist = pytz.timezone('Asia/Kolkata')
                local_time = timezone.now().astimezone(ist)

                # Save user info in session
                request.session['user_id'] = user.id
                request.session['user_name'] = user.name
                request.session['user_image'] = user.image.url  # image URL
                request.session['login_time'] = local_time.strftime('%I:%M:%S %p')

                return redirect('user_homepage')
            else:
                msg = "Your account is not activated yet."
        except RegisteredUser.DoesNotExist:
            msg = "Invalid credentials."

    return render(request, 'user_login.html', {'msg': msg})

def admin_login(request):
    msg = ''
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        # Define admin credentials
        admins = {
            'chandu': 'chandu',
            'abhishek': 'abhishek',
            'rahul': 'rahul',
            'sanjana': 'sanjana'
        }

        if name in admins and admins[name] == password:
            return redirect('admin_home')
        else:
            msg = "Invalid admin credentials."

    return render(request, 'admin_login.html', {'msg': msg})

def admin_home(request):
    return render(request, 'admin_home.html')
    
def admin_dashboard(request):
    users = RegisteredUser.objects.all()
    return render(request, 'admin_dashboard.html', {'users': users})

def activate_user(request, user_id):
    user = RegisteredUser.objects.get(id=user_id)
    user.is_active = True
    user.save()
    return redirect('admin_dashboard')

def deactivate_user(request, user_id):
    user = RegisteredUser.objects.get(id=user_id)
    user.is_active = False
    user.save()
    return redirect('admin_dashboard')

def delete_user(request, user_id):
    user = RegisteredUser.objects.get(id=user_id)
    user.delete()
    return redirect('admin_dashboard')



def home(request):
    return render(request, 'home.html')

def user_homepage(request):
    if 'user_id' not in request.session:
        # User not logged in, redirect to login page
        return redirect('user_login')

    user_name = request.session.get('user_name')
    user_image = request.session.get('user_image')
    login_time = request.session.get('login_time')

    context = {
        'user_name': user_name,
        'user_image': user_image,
        'login_time': login_time,
    }
    return render(request, 'users/user_homepage.html', context)

def user_logout(request):
    request.session.flush()  # Clears all session data
    return redirect('user_login')



import random
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.contrib import messages
from .models import RegisteredUser

otp_storage = {}  # Temporary dictionary to store OTPs

def send_otp(email):
    otp = random.randint(100000, 999999)  # Generate a 6-digit OTP
    otp_storage[email] = otp

    subject = "Password Reset OTP"
    message = f"Your OTP for password reset is: {otp}"
    from_email = "saikumardatapoint1@gmail.com"  # Change this to your email
    send_mail(subject, message, from_email, [email])

    return otp

def forgot_password(request):
    if request.method == "POST":
        email = request.POST.get("email")

        if RegisteredUser.objects.filter(email=email).exists():
            send_otp(email)
            request.session["reset_email"] = email  # Store email in session
            return redirect("verify_otp")
        else:
            messages.error(request, "Email not registered!")

    return render(request, "forgot_password.html")

def verify_otp(request):
    if request.method == "POST":
        otp_entered = request.POST.get("otp")
        email = request.session.get("reset_email")

        if otp_storage.get(email) and str(otp_storage[email]) == otp_entered:
            return redirect("reset_password")
        else:
            messages.error(request, "Invalid OTP!")

    return render(request, "verify_otp.html")

def reset_password(request):
    if request.method == "POST":
        new_password = request.POST.get("new_password")
        email = request.session.get("reset_email")

        if RegisteredUser.objects.filter(email=email).exists():
            user = RegisteredUser.objects.get(email=email)
            user.password = new_password  # Updating password
            user.save()
            messages.success(request, "Password reset successful! Please log in.")
            return redirect("user_login")

    return render(request, "reset_password.html")



# -------------------------ML CODE------------------------
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from django.shortcuts import render
from django.conf import settings


def train_models(request):
    # Load dataset
    file_path = os.path.join(settings.MEDIA_ROOT, 'Hyderabad Metro Dataset.csv')
    df = pd.read_csv(file_path, parse_dates=["DateTime"])
    df = df.dropna()
    df['TicketsBooked'] = df['TicketsBooked'].astype(float)
    df.sort_values("DateTime", inplace=True)

    # Time series data
    df_grouped = df.groupby("DateTime")["TicketsBooked"].sum()
    ts_train = df_grouped[:-5]
    ts_test = df_grouped[-5:]

    # Train ARIMA
    arima_model = ARIMA(ts_train, order=(1, 1, 0))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=5)
    arima_mae = mean_absolute_error(ts_test, arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(ts_test, arima_forecast))

    # Save ARIMA model
    arima_model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'arima_model.pkl')
    joblib.dump(arima_fit, arima_model_path)

    # Feature Engineering for ML models
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Weather'] = df['Weather'].map({'Clear': 0, 'Cloudy': 1, 'Rainy': 2})
    features = ['Hour', 'DayOfWeek', 'IsHoliday', 'Weather']
    X = df[features]
    y = df['TicketsBooked']
    X_train, X_test = X.iloc[:-5], X.iloc[-5:]
    y_train, y_test = y.iloc[:-5], y.iloc[-5:]

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    joblib.dump(rf_model, os.path.join(settings.MEDIA_ROOT, 'models', 'rf_model.pkl'))

    # Train XGBoost
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    joblib.dump(xgb_model, os.path.join(settings.MEDIA_ROOT, 'models', 'xgb_model.pkl'))

    # Save residuals, trend, seasonal
    result = seasonal_decompose(ts_train, model='additive', period=24)
    for comp, label in zip([result.resid, result.trend, result.seasonal, ts_train],
                           ['residual', 'trend', 'seasonal', 'original']):
        plt.figure(figsize=(10, 4))
        plt.plot(comp)
        plt.title(f"{label.capitalize()} Component")
        plt.tight_layout()
        plt.savefig(os.path.join(settings.MEDIA_ROOT, 'plots', f"{label}.png"))
        plt.close()

    # ACF & PACF
    plot_acf(ts_train.dropna(), lags=40)
    plt.tight_layout()
    plt.savefig(os.path.join(settings.MEDIA_ROOT, 'plots', 'acf.png'))
    plt.close()

    plot_pacf(ts_train.dropna(), lags=40)
    plt.tight_layout()
    plt.savefig(os.path.join(settings.MEDIA_ROOT, 'plots', 'pacf.png'))
    plt.close()

    # Daily Plot by Week
    df['Weekday'] = df['DateTime'].dt.day_name()
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Weekday', y='TicketsBooked', data=df)
    plt.title("Daily Plot by Week")
    plt.tight_layout()
    plt.savefig(os.path.join(settings.MEDIA_ROOT, 'plots', 'daily_by_week.png'))
    plt.close()

    # Monthly Plot
    df['Month'] = df['DateTime'].dt.month
    monthly = df.groupby('Month')["TicketsBooked"].sum()
    monthly.plot(kind='bar', figsize=(8, 4), title="Monthly Rides")
    plt.tight_layout()
    plt.savefig(os.path.join(settings.MEDIA_ROOT, 'plots', 'monthly.png'))
    plt.close()

    # Pass metrics to template
    context = {
        'results': {
            'ARIMA': {'MAE': round(arima_mae, 2), 'RMSE': round(arima_rmse, 2)},
            'RF': {'MAE': round(rf_mae, 2), 'RMSE': round(rf_rmse, 2)},
            'XGB': {'MAE': round(xgb_mae, 2), 'RMSE': round(xgb_rmse, 2)},
        },
        'plots': [
            'original.png', 'residual.png', 'trend.png', 'seasonal.png',
            'acf.png', 'pacf.png', 'daily_by_week.png', 'monthly.png'
        ]
    }
    return render(request, 'users/results.html', context)


# from .forms import TicketPredictionForm
# from django.conf import settings
# import joblib
# import os
# import numpy as np
# from django.shortcuts import render

# def predict_view(request):
#     prediction = None

#     if request.method == 'POST':
#         form = TicketPredictionForm(request.POST)
#         if form.is_valid():
#             dt = form.cleaned_data['datetime_input']
#             station = form.cleaned_data['station']
#             hour = dt.hour
#             day = dt.weekday()
#             holiday = int(form.cleaned_data['is_holiday'])
#             weather = int(form.cleaned_data['weather'])

#             input_features = np.array([[hour, day, holiday, weather]])

#             # Load models
#             rf_model = joblib.load(os.path.join(settings.MEDIA_ROOT, 'models/rf_model.pkl'))
#             xgb_model = joblib.load(os.path.join(settings.MEDIA_ROOT, 'models/xgb_model.pkl'))

#             # Predict
#             rf_pred = rf_model.predict(input_features)[0]
#             xgb_pred = xgb_model.predict(input_features)[0]

#             prediction = {
#                 'station': station,
#                 'datetime': dt.strftime('%Y-%m-%d %H:%M'),
#                 'rf': int(round(rf_pred)),
#                 'xgb': int(round(xgb_pred)),
#             }
#     else:
#         form = TicketPredictionForm()

#     return render(request, 'users/predict_form.html', {'form': form, 'prediction': prediction})


from .forms import TicketPredictionForm
from django.conf import settings
import joblib
import os
import numpy as np
from django.shortcuts import render
from datetime import datetime

def predict_view(request):
    prediction = None

    if request.method == 'POST':
        form = TicketPredictionForm(request.POST)
        if form.is_valid():
            # Get inputs from form
            date_input = form.cleaned_data['date_input']
            hour = int(form.cleaned_data['hour'])
            station = form.cleaned_data['station']
            holiday = int(form.cleaned_data['is_holiday'])
            weather = form.cleaned_data['weather']

            # Convert weather to numeric
            weather_map = {'Clear': 0, 'Cloudy': 1, 'Rainy': 2}
            weather_num = weather_map[weather]

            # Combine date + hour into datetime
            dt = datetime.combine(date_input, datetime.min.time()).replace(hour=hour)
            day = dt.weekday()

            # Prepare features
            input_features = np.array([[hour, day, holiday, weather_num]])

            # Load models
            rf_model = joblib.load(os.path.join(settings.MEDIA_ROOT, 'models/rf_model.pkl'))
            xgb_model = joblib.load(os.path.join(settings.MEDIA_ROOT, 'models/xgb_model.pkl'))

            # Predict
            rf_pred = rf_model.predict(input_features)[0]
            xgb_pred = xgb_model.predict(input_features)[0]

            # Prepare output
            prediction = {
                'station': station,
                'datetime': dt.strftime('%Y-%m-%d %H:%M'),
                'rf': int(round(rf_pred)),
                'xgb': int(round(xgb_pred)),
            }
    else:
        form = TicketPredictionForm()

    return render(request, 'users/predict_form.html', {'form': form, 'prediction': prediction})
