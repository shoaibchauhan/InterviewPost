# Generated by Django 3.2 on 2024-11-07 15:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Filee',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('filename', models.CharField(max_length=250, unique=True)),
                ('filesummary', models.TextField()),
            ],
        ),
    ]
