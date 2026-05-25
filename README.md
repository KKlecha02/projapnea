# Apnea Detector

Aplikacja webowa do wykrywania epizodów bezdechu sennego na podstawie danych SpO2.

## Co robi

Algorytm wykrywa spadki saturacji krwi o co najmniej 3% lub wartości poniżej 90% trwające minimum 10 sekund. Na tej podstawie oblicza wskaźnik AHI i określa nasilenie bezdechu. Wynik jest pokazywany na wykresie SpO2 z zaznaczonymi epizodami.

## Uruchomienie

    git clone <url-repozytorium>
    cd apnea-app
    pip install -r requirements.txt
    flask --app src/app run

Następnie otwórz http://127.0.0.1:5000 w przeglądarce.

## Dane

Dane pacjentów powinny być w folderze ../Data/ względem katalogu projektu. Ścieżki można zmienić w config.json.

Podział danych:
- Trening: pacjenci 01-24
- Walidacja: pacjenci 25-30 i 50

## Walidacja modelu

    python validate.py

Wyniki są zapisywane do docs/validation_report.json i docs/confusion_matrix.png.

## Trenowanie modelu

    python train.py

Wymaga PyTorch. Zapisuje wytrenowany model do model.pt.

## Testy

    pytest tests/

## Struktura projektu

    src/
      apnea.py          - algorytm detekcji (reguły)
      apnea_engine.py   - model CNN-LSTM
      app.py            - serwer Flask
    tests/
      test_apnea.py
      test_integration.py
    data/
      input.json        - przykładowe dane wejściowe
      expected_output.json
    docs/
      architecture.md
      api.md
    results/            - historia analiz
    templates/
      index.html
      result.html
    static/
      style.css
    load_dataset.py
    validate.py
    train.py
    config.json
    wsgi.py
    requirements.txt

## Deploy na PythonAnywhere

1. Załóż konto na pythonanywhere.com

2. Otwórz konsolę Bash i wklej:

        git clone <url-repozytorium>
        cd apnea-app

3. Utwórz wirtualne środowisko:

        mkvirtualenv apnea --python=python3.10
        pip install -r requirements.txt

4. W zakładce Web kliknij "Add a new web app", wybierz Manual configuration i Python 3.10

5. W sekcji Code ustaw:
   - Source code: /home/<username>/apnea-app
   - Working directory: /home/<username>/apnea-app
   - W pliku WSGI zastąp zawartość tym:

          import sys
          sys.path.insert(0, '/home/<username>/apnea-app')
          from src.app import app as application

6. W sekcji Virtualenv wpisz: /home/<username>/.virtualenvs/apnea

7. Utwórz folder na wyniki:

        mkdir ~/apnea-app/results

8. Kliknij Reload i wejdź na <username>.pythonanywhere.com
