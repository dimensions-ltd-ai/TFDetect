import setuptools

with open('requirements.txt') as f:
    reqs = []
    for line in f:
        line = line.strip()
        reqs.append(line.split('==')[0])

setuptools.setup(
    name='TFDetect',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='',
    license='',
    author='Hamza Naeem',
    author_email='dimensionsltdai@gmail.com',
    description='tensorflow object detection api',
    install_requires=reqs
)
