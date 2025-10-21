from django.contrib import admin
from .models import RegisteredUser

@admin.register(RegisteredUser)
class RegisteredUserAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'mobile', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('name', 'email', 'mobile')
