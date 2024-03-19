.. -*- coding: utf-8 -*-
.. sigmoid_docs documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

****************************************
SIGMOID
****************************************

:Release: |release|
:Date: |today|

SIGMOID is an AutoML framework that enables you to generate, train and deploy scalable supervised machine learning pipelines.

Our goal is

Sigmoid works with a variety of data types such as numbers, dates, categories, tags, text, arrays and various multimedia formats.
These data types can be combined together to solve complex problems.

For details as to how sigmoid works, check out the "Sigmoid Philosophy" page.

Quick Guide
=======================
- :ref:`Installation <Installation>`
- :ref:`Contribute to Sigmoid <Contribute to Sigmoid>`

Installation
============

You can install Sigmoid as follows:

.. code-block:: bash

   git clone git@github.com:mindsdb/sigmoid
   cd sigmoid/
   pip install -r requirements.txt
   pip install -e .

.. note:: depending on your environment, you might have to use pip instead of pip3 in the above command.

However, we recommend creating a python virtual environment

.. code-block:: bash

   mkdir sigmoid-env
   python -m venv ./sigmoid-env/
   source sigmoid-env/bin/activate

Setting up a dev environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Clone sigmoid
- Run ``cd sigmoid && pip install -r requirements.txt`` and ``pip install -e .``
- Check that the unit-tests are passing by going into the directory where you cloned sigmoid and running: ``python -m unittest discover tests``

.. warning:: If ``python`` default to python2.x on your environment use ``python3`` and ``pip3`` instead

Setting up a VSCode environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Install and enable setting sync using github account (if you use multiple machines)
* Install pylance (for types) and make sure to disable pyright
* Go to ``Python > Lint: Enabled`` and disable everything *but* flake8
* Set ``python.linting.flake8Path`` to the full path to flake8 (which flake8)
* Set ``Python › Formatting: Provider`` to autopep8
* Add ``--global-config=<path_to>/sigmoid/.flake8`` and ``--experimental`` to ``Python › Formatting: Autopep8 Args``
* Install live share and live share whiteboard


Contribute to Sigmoid
=======================

We love to receive contributions from the community and hear your opinions! We want to make contributing to `sigmoid` as easy as it can be.

Being part of the core Sigmoid team is possible to anyone who is motivated and wants to be part of that journey!

Please continue reading this guide if you are interested in helping democratize machine learning.

How can you help us?
^^^^^^^^^^^^^^^^^^^^^^^^
* Report a bug
* Improve documentation
* Solve an issue
* Propose new features
* Discuss feature implementations
* Submit a bug fix
* Test Sigmoid with your own data and let us know how it went!

Code contributions
^^^^^^^^^^^^^^^^^^^^^^^^
In general, we follow the `fork-and-pull <https://docs.github.com/en/github/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model>`_ git workflow. Here are the steps:

1. Fork the Sigmoid repository
2. Checkout the ``staging`` branch, which is the development version that gets released weekly (there can be exceptions, but make sure to ask and confirm with us).
3. Make changes and commit them
4. Make sure that the CI tests pass. You can run the test suite locally with ``flake8 .`` to check style and ``python -m unittest discover tests`` to run the automated tests. This doesn't guarantee it will pass remotely since we run on multiple envs, but should work in most cases.
5. Push your local branch to your fork
6. Submit a pull request from your repo to the ``staging`` branch of ``mindsdb/open-sigmoid`` so that we can review your changes. Be sure to merge the latest from staging before making a pull request!

.. note:: You will need to sign a CLI agreement for the code since sigmoid is under a GPL license!


Feature and Bug reports
^^^^^^^^^^^^^^^^^^^^^^^^
We use GitHub issues to track bugs and features. Report them by opening a `new issue <https://github.com/mindsdb/open-sigmoid/issues/new/choose>`_ and fill out all of the required inputs.


Code review process
^^^^^^^^^^^^^^^^^^^^^^^^^
Pull request (PR) reviews are done on a regular basis. **If your PR does not address a previous issue, please make an issue first**.

If your change has a chance to affecting performance we will run our private benchmark suite to validate it.

Please, make sure you respond to our feedback/questions.


Community
^^^^^^^^^^^^^^^^^^^^^^^^^
If you have additional questions or you want to chat with MindsDB core team, you can join our community:

.. raw:: html

    <embed>
    <a href="https://join.slack.com/t/mindsdbcommunity/shared_invite/zt-o8mrmx3l-5ai~5H66s6wlxFfBMVI6wQ" target="_blank"><img src="https://img.shields.io/badge/slack-@mindsdbcommunity-blueviolet.svg?logo=slack " alt="MindsDB Community"></a>
    </embed>

To get updates on Sigmoid and MindsDB’s latest announcements, releases, and events, sign up for our `Monthly Community Newsletter <https://mindsdb.com/newsletter/?utm_medium=community&utm_source=github>`_.

Join our mission of democratizing machine learning and allowing developers to become data scientists!

Contributor Code of Conduct
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Please note that this project is released with a `Contributor Code of Conduct <https://github.com/mindsdb/sigmoid/blob/stable/CODE_OF_CONDUCT.md>`_. By participating in this project, you agree to abide by its terms.


Current contributors
=======================

.. raw:: html

    <embed>
    <a href="https://github.com/mindsdb/open-sigmoid/graphs/contributors">
      <img src="https://contributors-img.web.app/image?repo=mindsdb/sigmoid" />
    </a>
    </embed>


License
=======================

| `Sigmoid License <https://github.com/mindsdb/sigmoid/blob/master/LICENSE>`_

Other Links
=======================
.. toctree::
   :maxdepth: 8

   tutorials
   sigmoid_philosophy
   data_brokers
   preprocessing
   auto_encoding
   api