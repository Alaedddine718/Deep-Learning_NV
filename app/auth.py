from __future__ import annotations
from flask import Blueprint, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash

bp = Blueprint('auth', __name__)
USERS = {}  # demo en memoria

@bp.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        if not email or not password:
            return "Faltan campos", 400
        if email in USERS:
            return "Usuario ya existe", 400
        USERS[email] = generate_password_hash(password)
        return redirect(url_for('login_page'))
    return "OK", 200

@bp.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        if email not in USERS or not check_password_hash(USERS[email], password):
            return "Credenciales inválidas", 400
        session['user'] = email
        return redirect(url_for('index'))
    return "OK", 200

@bp.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))
