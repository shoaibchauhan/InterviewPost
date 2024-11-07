# FastAPI and Django Project with MongoDB and Predibase Integration

## Overview
This project is a web application built using FastAPI and Django, with MongoDB as the database. It includes functionality to handle file uploads, generate summaries using Predibase, and perform CRUD operations.

## Features
- **FastAPI** for handling API requests
- **Django** for the admin interface and ORM
- **Pydantic** for the view
- **MongoDB** for database management
- **Predibase** for generating summaries
- **pypdf,python-docx, python-pptx, PyMuPDF, pypdf2** for file format
- Supports file uploads and processing for `.docx`, `.pptx`, and `.pdf` files
- API methods:
  - POST: Upload and process files
  - GET by ID: Retrieve a specific record
  - GET all: Retrieve all records

## Requirements
- Python 3.7+
- MongoDB
- 

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shoaibchauhan/InterviewPost.git
   cd InterviewPost
   
2. Activate virtual environment:
3. Install dependencies from requirement.txt
4. Setup your database in settings.py (PostgreSQL)
5. Add your Predibase API token in the FastAPI application code:
6. Run Django migrations to set up your database:
7. Start the FastAPI server:
