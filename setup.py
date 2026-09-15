from setuptools import find_packages, setup,find_packages
our_p="-e ."
def get_req(file_path):
    with open(file_path) as f:
        req = f.readlines()
        req = [re.replace("\n","") for re in req]
        if our_p in req:
            req.remove(our_p)
    return req


setup(
    name='roc',
    version='0.1',
    author='rahul',
    author_email='rahulrc2030@gmail.com',
    install_requires=get_req("req.txt"),
    packages=find_packages()
)