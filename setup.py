from setuptools import find_packages, setup

FILE_NAME = 'requirements.txt'
HYPHEN_E_DOT ='-e .'

def get_requirements(File_name):
    with open(File_name) as requires:
        requirements = [req.replace('\n','') for req in requires]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name = 'APS_FAULT_DETECTION_ML_PROJET',
    author = 'SAKSHIT_ATTRI',
    author_email = 'sakshit2000@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements(File_name = FILE_NAME)
)