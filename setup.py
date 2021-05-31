from setuptools import setup
from setuptools import find_packages


setup(name='clip_mania',
      version='0.0.1',
      description='Custom training with OpenAI CLIP; classification tasks; zero-shot examples; and a fully '
                  'dockerised web-service.',
      author='Wilder Rodrigues',
      author_email='wilder.rodrigues@gmail.com',
      url='git@github.com:artificially-ai/clip-mania.git',
      install_requires=[
          'numpy',
          'pandas',
          'torch',
          'torchvision',
          'scikit-image',
          'ftfy',
          'regex',
          'tqdm',
          'pillow',
          'pytest',
          'git+https://github.com/openai/CLIP.git'],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.8',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      python_requires='>=3.8',
      packages=find_packages())
