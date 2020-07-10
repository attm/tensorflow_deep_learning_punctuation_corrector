from setuptools import setup

setup(
   name='text_punctuation_corrector',
   version='0.1',
   description='Simple tf.keras punctuation corrector, made with encoder-decoder model',
   author='atgm1113',
   author_email='atgm1113@gmail.com',
   packages=[  "corrector_dataset_builder",
               "encdec_model",
               "encdec_model_builder",
               "flask_app",
               "tests",
               "utils",
               "config",
               "main"]
)