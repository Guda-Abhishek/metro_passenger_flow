
# from django import forms
# import datetime

# class TicketPredictionForm(forms.Form):
#     STATION_CHOICES = [
#         ('Wakad', 'Wakad'),
#         ('Kothrud', 'Kothrud'),
#         ('Shivajinagar', 'Shivajinagar'),
#         ('Hinjewadi', 'Hinjewadi'),
#         ('Deccan', 'Deccan'),
#         ('Swargate', 'Swargate'),
#         ('Pimpri', 'Pimpri')
#     ]

#     WEATHER_CHOICES = [(0, 'Clear'), (1, 'Cloudy'), (2, 'Rainy')]
#     HOLIDAY_CHOICES = [(0, 'No'), (1, 'Yes')]

#     station = forms.ChoiceField(choices=STATION_CHOICES)
    
#     datetime_input = forms.DateTimeField(
#         label="Future Date & Time",
#         input_formats=['%Y-%m-%dT%H:%M'],  # Important
#         widget=forms.DateTimeInput(
#             attrs={'type': 'datetime-local'},
#             format='%Y-%m-%dT%H:%M'
#         ),
#         initial=datetime.datetime.now
#     )

#     is_holiday = forms.ChoiceField(choices=HOLIDAY_CHOICES)
#     weather = forms.ChoiceField(choices=WEATHER_CHOICES)



from django import forms
import datetime

class TicketPredictionForm(forms.Form):
    STATION_CHOICES = [
        ('Tarnaka', 'Tarnaka'),
        ('Hitec City', 'Hitec City'),
        ('Kukatpally', 'Kukatpally'),
        ('Ameerpet', 'Ameerpet'),
        ('LB Nagar', 'LB Nagar'),
        ('Uppal', 'Uppal'),
        ('Secunderabad East', 'Secunderabad East'),
        ('Hyderabad Central', 'Hyderabad Central'),
        ('Miyapur', 'Miyapur'),
        ('Dilsukhnagar', 'Dilsukhnagar'),
    ]

    WEATHER_CHOICES = [
        ('Clear', 'Clear'),
        ('Rainy', 'Rainy'),
        ('Cloudy', 'Cloudy'),
    ]

    HOLIDAY_CHOICES = [(0, 'No'), (1, 'Yes')]

    station = forms.ChoiceField(choices=STATION_CHOICES)

    # Date only (future day selection)
    date_input = forms.DateField(
        label="Future Date",
        widget=forms.DateInput(
            attrs={'type': 'date'}
        ),
        initial=datetime.date.today
    )

    # Hour selection (6 to 23)
    HOUR_CHOICES = [(h, f"{h}:00") for h in range(6, 24)]
    hour = forms.ChoiceField(choices=HOUR_CHOICES)

    is_holiday = forms.ChoiceField(choices=HOLIDAY_CHOICES)
    weather = forms.ChoiceField(choices=WEATHER_CHOICES)
