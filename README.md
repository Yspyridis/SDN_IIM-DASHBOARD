[![GitHub issues](https://img.shields.io/github/issues/Yspyridis/SDN_IIM-DASHBOARD)](https://github.com/Yspyridis/SDN_IIM-DASHBOARD/issues)
[![GitHub license](https://img.shields.io/github/license/Yspyridis/SDN_IIM-DASHBOARD)](https://github.com/Yspyridis/SDN_IIM-DASHBOARD)


# IIM Dashboard

The IIM Dashboard is a tool developed to display and manage the islanding schemes provided by the IIM component for the purposes of the SDN-microSENSE project.

With the instructions below, you can setup the project on a local machine for testing purposes.

## Installation

Clone the repository and create a Python virtual environment:

```
git clone https://github.com/Yspyridis/SDN_IIM-DASHBOARD.git
python3 -m venv venv
```

Activate the environment and install the requirements:

```
source venv/bin/activate 
cd iim_dashboard
pip3 install -r requirements.txt
pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Setup the database (for Ubuntu):
```
sudo apt-get install postgresql
sudo su - postgres
createuser -P iim
createdb -O iim iimdb
```

Migrate the project:
```
python3 manage.py makemigrations
python3 manage.py migrate
```

If there is a migration error, drop the database:
```
dropdb iimdb
```

You may also need:
```
find . -path "*/migrations/*.py" -not -name "__init__.py" -delete
find . -path "*/migrations/*.pyc"  -delete
```

Run the server:
```
python3 manage.py runserver
```
