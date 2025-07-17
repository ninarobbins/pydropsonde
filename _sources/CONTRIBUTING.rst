Contributing
============

Firstly, thanks for thinking of contributing to ``pydropsonde``; we are hoping that you didn't land here by mistake. But even if you did, now that you are here why not help us out a bit with this package? There are `different ways <how-help-us_>`_ in which you can contribute to pydropsonde. To know how to get started, jump to :ref:`dev_workflow`.

.. _how-help-us:

How can you help us?
--------------------

You can help us by:

- Submitting issues
  - Tell us about any bugs you encounter, enhancements or features that you think the package should include, or even things like some missing documentation that might be useful to you and others. Simply raise an `Issue <https://github.com/atmdrops/pydropsonde/issues>`_! Even if you are not sure if something qualifies as an issue, just raise an issue anyway. Worst case, someone will point out the solution or you might figure it out yourself and help out someone in the future facing the same problem. It's a win-win.
- Documenting
  - A large and very important part of ``pydropsonde`` is the documentation that goes with it. This includes the protocols for operations and data processing as well as the explanations behind them. We also need how-to guides to explain better how the package can be employed. And there are functions and classes that can benefit from better docstrings. In fact, this very document ``CONTRIBUTING.md`` needs more documentation. There is almost never enough documentation. So, pick out a small aspect of the ``pydropsonde`` docs and try your hand at it. The universe will thank you. Just try it.
- Coding
  - Scripts, classes, methods, tests... A lot remains to be done in the package.

Have a look at the `Issues <https://github.com/atmdrops/pydropsonde/issues>`_ and you will find where ``pydropsonde`` needs help. Pick an issue and assign it to yourself, so others know that you are working on that. If you are not sure how to proceed, you can express your interest by commenting on the issue, and someone should help you out.

.. _dev_workflow:

Development Workflow
--------------------

1. **Fork & clone pydropsonde**

    Fork the `pydropsonde <https://github.com/atmdrops/pydropsonde.git>`_ repository.

    Clone your fork and set the original repository to remote upstream with the following commands:

    .. code-block:: bash

        git clone git@github.com:<your-github-username>/pydropsonde.git
        cd pydropsonde
        git remote add upstream git@github.com:atmdrops/pydropsonde.git

2. **Create the development environment**

    ``pydropsonde`` is developed in Python. The packages needed for development is specified in the `environment.yaml` file. The package manager `conda <https://conda.io/>`_ can be used to create an environment from the file.
    Be sure you are in the ``pydropsonde`` home directory, i.e. the location of the `pyproject.toml` file. This will ensure the ``pip`` editable installation of the ``pydropsonde`` package within the environment itself.

    So, if you are in the right directory, let's create the environment.

    .. code-block:: bash

        conda env create -f environment.yml

    Activate the environment with:
    .. code-block:: bash

       conda activate pydropsonde_env


    **_Get pre-commit working for you_**

    If you created the environment with the ``environment.yaml`` file, then ``pre-commit`` should already be present in your environment. ``pre-commit`` is used to employ hooks for checking (and in some cases, fixing) the code before commits are made. To get pre-commit to check automatically every time you commit, use the following command:

    .. code-block:: bash

        pre-commit install


    That's it.  ``pre-commit`` will now parse through all hooks in the ``.pre-commit-config.yaml`` file and do its thing accordingly.

    Every time there is a change in the config YAML file, be sure to apply those changes for all existing files too with the following command:

    .. code-block:: bash

        pre-commit run --all-files


3. **Create a branch**

    Now we have a local copy of our fork and we have the environment ready to start developing.

    It is always good coding practice to work on a different branch every time you start working on a new feature / bug-fix (yes, despite having your own fork).

    Create a branch and checkout to start working on it.

    .. code-block:: bash

        git branch my-new-feature
        git checkout my-new-feature


4. **Make your changes**

    Do your edits and push to your fork. Behold git's holy trinity!

    .. code-block:: bash

      git add . # will add all uncommitted changes
      git commit -m "your commit message here" # consider giving a detailed message & not simply a header
      git push # for the first push of a branch, track it e.g. git push -u origin my-new-feature


    Every commit makes changes that are justified by one reason. The size of a commit could be a single character change or a change in thousands of lines across millions of files. But the reason behind the commit should ideally be as solitary as possible. Commit often, but not too often. Henry VIII said that.

    For making changes to the documentation, refer the :ref:`dokudev` section for steps.

5. **Submit pull request**

    Head over to Github and from the relevant branch in your fork, create a `Pull Request (PR) <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_.

    You can `request a PR review <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review>`_ from someone. They will help with some feedback or might wholeheartedly agree with your changes. Others might also comment with their opinion. Add any necessary changes with subsequent commits to the same branch. Once everyone involved in this conversation is satisfied, the PR is merged. From personal experience though, frantically refreshing the browser every three seconds doesn't speed up the response rate of others. Sometimes, PRs can take their own sweet time.

6. **... And that's it! Thanks for helping**

Adding dependencies
-------------------

Currently, pyDropsonde is build with `poetry <https://python-poetry.org/>`_. If you need a new package for your contribution, please remember to
 - add it to the ``pyproject.toml``
 - create a new ``poetry.lock`` (see the `poetry descriptions <https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock>`_)
 - add it to the ``environment.yaml``



.. _dokudev:

Documentation development
-------------------------

``pydropsonde`` uses `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ with the `Book theme <https://sphinx-book-theme.readthedocs.io/en/stable/>`_ to display its documentation, and is hosted by `Github pages <https://pages.github.com/>`_. All documentation comes from rsStructuredText  files or Jupyter notebooks, except the API reference, which is built automatically from the docstrings in the modules, thanks to `sphinx-autosummary <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_.

Steps to make documentation changes
-----------------------------------

1. You'll find the source files for documentation in the ``docs/source/`` directory. If you want to change documentation in the API reference, then head over to the corresponding module in the source code and change the relevant docstring.

2. Make the change. Here are some referencing tips for both Markdown files and for docstrings.

   - For cross-referencing within the document, use
      .. code-block::

          :ref:`section_label`

      for example, ``:ref:`dokudev``` in the source code will give you the following link: :ref:`dokudev`. This works because the section :ref:`dokudev` is tagged with the label ``dokudev`` in the source code, which is done by adding the line ``.. _dokudev:`` at the beginning of the section.
      If you want to add some custom text to your reference, you can do so by adding the text in the following way: ```New Fancy Text <dokudev_>`_``. This will give you the following link: `New Fancy Text <dokudev_>`_, i.e. the text ``New Fancy Text`` will be displayed, but the link will still point to the section with the label ``dokudev``.


   - For cross-referencing a different document, use
      .. code-block::

        :doc:`description <path/to/file>`

      for example, ``:doc:`landing <index>``` will give you the following link: :doc:`landing <index>`. This works because the file ``index.rst`` is in the same directory as the file you are referencing from. If the file is in a different directory, you need to specify the path to the file relative to the current file.


   - For URLs
      .. code-block::

          `description <url>`_

      for example, ```github <https://github.com/>`_`` will give you the following link: `github <https://github.com/>`_. Note that the URL will open in the same tab.





3. Rebuild the documentation with:

.. code-block:: bash

    sphinx-build -n docs/source docs/_build


The `-n` flag is to enable `nitpicky mode <https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpicky>`_, so that we catch all warnings with missing references.

When you open a pull request and merge into the `main` branch, the documentation will be built automatically and deployed to https://atmdrops.github.io/pydropsonde/.
