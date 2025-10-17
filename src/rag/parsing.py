import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)



import json
import re


# ============================================================
# 1. Exemple de parser personnalisé (tu peux le modifier selon ton besoin)
# ============================================================


def mega_json_corrector(raw_json_text):
    # 1. Nettoyage des simples guillemets par doubles (pour JSON)
    cleaned = raw_json_text.replace("'", '"')
    cleaned = raw_json_text.replace("`", '"')

    # 2. Suppression des doubles parenthèses vides incorrectes, ex : sentiment()
    cleaned = re.sub(r'\(\)', '""', cleaned)

    print(f"\n[DEBUG] ETAPE Suppression parenthèse : {cleaned}\n")



    # 3. Correction : ajouter des guillemets autour de clés non entre guillemets dans les fonctions
    # Exemple : gt(sentiment, 0.8) => gt(/"sentiment/", 0.8)

    pattern_keys = re.compile(r'(\w+)\(\s*("[^"]*"|[^()\s,]+)\s*([,)])')

    cleaned = pattern_keys.sub(repl_keys, cleaned)

    print(f"\n[DEBUG] ETAPE Ajout '\"' : \n{cleaned}\n")




    # 3. Vérification et ajout de guillemets autour des valeurs dates (format simple)
    cleaned = escape_dates_and_datetimes(cleaned)
    cleaned = escape_date_formats(cleaned)

    print(f"\n[DEBUG] ETAPE Ajout parenthèse date : {cleaned}\n")


    # 5. Équilibrage simple des parenthèses (ajout parenthèse fermante si manquante)
    open_par = cleaned.count('(')
    close_par = cleaned.count(')')
    if open_par > close_par:
        cleaned += ')' * (open_par - close_par)
    elif close_par > open_par:
        cleaned = '(' * (close_par - open_par) + cleaned

    print(f"\n[DEBUG] ETAPE 5 : {cleaned}\n")
    # 6. Suppression nombre string 

    # Remplace les valeurs numériques dans les chaînes guillemetées
    cleaned = re.sub(r'"\s*[<>]=?\s*([^"]+)"', convert_numeric, cleaned)

    print(f"\n[DEBUG] ETAPE 6 : {cleaned}\n")


    # 7. Essai de parsing JSON sur le filtre transformé
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        # En cas d'échec, lever une erreur détaillée
        raise ValueError(f"Parsing JSON impossible après corrections. Erreur: {e}\n[TEST] Sortie :\n{cleaned}")






def is_number_like(s):
    pattern = re.compile(r'^[<>]=?\s*[-+]?\d*\.?\d+(e[-+]?\d+)?$')
    s_clean = s.replace(" ", "")
    return bool(pattern.match(s_clean))

def repl_keys(m):
    func = m.group(1)
    arg = m.group(2).strip()
    sep = m.group(3) if m.lastindex >= 3 else ""

    # Si l'argument ressemble à une fonction imbriquée, ne rien modifier
    if re.match(r'^\w+\(.*\)$', arg):
        return m.group(0)

    # Si l'argument est déjà bien guillemeté, ne rien modifier
    if arg.startswith('"') and arg.endswith('"'):
        return f"{func}({arg}{sep}"

    # Si argument ressemblant à un nombre/comparateur, ne rien modifier
    if is_number_like(arg):
        return f"{func}({arg}{sep}"

    # Sinon, on nettoie l'argument pour ne garder que texte brut
    cleaned_arg = arg.replace('"', '').replace('\\', '')

    # Retourne la chaîne avec guillemets échappés mais en conservant la structure d'origine
    return f'{func}(\\"{cleaned_arg}\\"{sep}'




def convert_numeric(match):
    s = match.group(1)
    print(f"\n[DEBUG] Conversion numérique, {s}\n\n")
    # Supprime tout sauf chiffres, '.', '+', '-', 'e', 'E' (et supprime aussi les backslashs)
    num_str = re.sub(r'[^0-9.+\-eE]', '', s.replace('\\', ''))
    try:
        # Essaie conversion int, sinon float
        if '.' in num_str or 'e' in num_str.lower():
            num = float(num_str)
        else:
            num = int(num_str)
        return str(num)
    except:
        return s  # retourne tel quel si conversion impossible


def escape_dates_and_datetimes(text):
    # Regex pour matcher date YYYY-MM-DD ou datetime YYYY-MM-DD HH:MM:SS,
    # avec guillemets simples, doubles ou sans guillemets, mais ignorer celles déjà entre \"
    pattern = r'(?<!\\")([\'\"]?)(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?)(\1)(?!\\")'

    def replacer(match):
        date_str = match.group(2)
        # Si la date est déjà entourée par des guillemets échappés, on ne change rien
        if match.start() >= 2 and text[match.start()-2:match.start()] == r'\\"' and \
           text[match.end():match.end()+2] == r'\\"':
            return match.group(0)
        # Sinon, entoure la date avec guillemets échappés
        return f'\\"{date_str}\\"'

    return re.sub(pattern, replacer, text)

def escape_date_formats(text):
    # Remplace les "YYYY-MM-DD" par \"YYYY-MM-DD\" uniquement si pas déjà échappé
    pattern = r'(?<!\\)"(YYYY-MM-DD)"'
    return re.sub(pattern, r'\\"YYYY-MM-DD\\"', text)

