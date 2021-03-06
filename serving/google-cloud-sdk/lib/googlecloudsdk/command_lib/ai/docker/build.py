# -*- coding: utf-8 -*- #
# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions required to interact with Docker to build images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import json
import os
import posixpath
import re
import textwrap

from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils
from googlecloudsdk.core import log
from six.moves import shlex_quote

_DEFAULT_HOME = "/home"
_DEFAULT_WORKDIR = "/usr/app"
_DEFAULT_SETUP_PATH = "./setup.py"
_DEFAULT_REQUIREMENTS_PATH = "./requirements.txt"
_AUTONAME_PREFIX = "cloudai-autogenerated"
_AUTOGENERATED_TAG_LENGTH = 16


def _IsVertexTrainingPrebuiltImage(image_name):
  """Checks whether the image is pre-built by Vertex AI training."""
  prebuilt_image_name_regex = (r"^(us|europe|asia)-docker.pkg.dev/"
                               r"vertex-ai/training/"
                               r"(tf|scikit-learn|pytorch|xgboost)-.+$")
  return re.fullmatch(prebuilt_image_name_regex, image_name) is not None


def _SitecustomizeRemovalEntry(is_prebuilt_image):
  """Returns a Dockerfile entry that removes `sitecustomize` if it's Vertex AI Training pre-built container images."""
  return "RUN rm -rf /var/sitecustomize" if is_prebuilt_image else ""


def _GenerateCopyCommand(from_path, to_path, comment=None):
  """Returns a Dockerfile entry that copies a file from host to container.

  Args:
    from_path: (str) Path of the source in host.
    to_path: (str) Path to the destination in the container.
    comment: (str) A comment explaining the copy operation.
  """
  cmd = "COPY {}\n".format(json.dumps([from_path, to_path]))

  if comment is not None:
    formatted_comment = "\n# ".join(comment.split("\n"))
    return "# {}\n{}".format(formatted_comment, cmd)

  return cmd


def _DependencyEntries(is_prebuilt_image=False,
                       requirements_path=None,
                       setup_path=None,
                       extra_requirements=None,
                       extra_packages=None,
                       extra_dirs=None):
  """Returns the Dockerfile entries required to install dependencies.

  Args:
    is_prebuilt_image: (bool) Whether the base image is pre-built and provided
      by Vertex AI.
    requirements_path: (str) Path that points to a requirements.txt file
    setup_path: (str) Path that points to a setup.py
    extra_requirements: (List[str]) Required dependencies to be installed from
      remote resource archives.
    extra_packages: (List[str]) User custom dependency packages to install.
    extra_dirs: (List[str]) Directories other than the work_dir required.
  """
  ret = ""
  pip_version = "pip3" if is_prebuilt_image else "pip"

  if setup_path is not None:
    ret += textwrap.dedent("""
        {}
        RUN {} install --no-cache-dir .
        """.format(
            _GenerateCopyCommand(
                setup_path,
                "./setup.py",
                comment="Found setup.py file, thus copy it to the docker container."
            ), pip_version))

  if requirements_path is not None:
    ret += textwrap.dedent("""
        {}
        RUN {} install --no-cache-dir -r ./requirements.txt
        """.format(
            _GenerateCopyCommand(
                requirements_path,
                "./requirements.txt",
                comment="Found requirements.txt file, thus to the docker container."
            ), pip_version))

  if extra_packages is not None:
    for extra in extra_packages:
      package_name = os.path.basename(extra)
      ret += textwrap.dedent("""
        {}
        RUN {} install --no-cache-dir {}
        """.format(
            _GenerateCopyCommand(extra, package_name), pip_version,
            shlex_quote(package_name)))

  if extra_requirements is not None:
    for requirement in extra_requirements:
      ret += textwrap.dedent("""
        RUN {} install --no-cache-dir --upgrade {}
        """.format(pip_version, shlex_quote(requirement)))

  if extra_dirs is not None:
    for directory in extra_dirs:
      ret += "\n{}\n".format(_GenerateCopyCommand(directory, directory))

  return ret


def _GenerateEntrypoint(package, is_prebuilt_image=False):
  """Generates dockerfile entry to set the container entrypoint.

  Args:
    package: (Package) Represents the main application copied to the container.
    is_prebuilt_image: (bool) Whether the base image is pre-built and provided
      by Vertex AI.

  Returns:
    A string with Dockerfile directives to set ENTRYPOINT
  """

  # Make it consistent with Online python package training that python3
  # has been installed for all prebuilt images and used by default
  python_command = "python3" if is_prebuilt_image else "python"

  # Needs to use json so that quotes print as double quotes, not single quotes.
  if package.python_module is not None:
    exec_str = json.dumps([python_command, "-m", package.python_module])
  else:
    _, ext = os.path.splitext(package.script)
    executable = [python_command] if ext == ".py" else ["/bin/bash"]
    exec_str = json.dumps(executable + [package.script])

  return "\nENTRYPOINT {}".format(exec_str)


def _PreparePackageEntry(package):
  """Returns the Dockerfile entries required to append at the end before entrypoint.

  Including:
  - copy the parent directory of the main executable into a docker container.
  - inject an entrypoint that executes a script or python module inside that
    directory.

  Args:
    package: (Package) Represents the main application copied to and run in the
      container.
  """
  parent_dir = os.path.dirname(package.script) or "."

  copy_code = _GenerateCopyCommand(
      parent_dir,
      parent_dir,
      comment="Copy the source directory into the docker container.")

  return "\n{}\n".format(copy_code)


def _MakeDockerfile(base_image,
                    main_package,
                    container_workdir,
                    container_home,
                    requirements_path=None,
                    setup_path=None,
                    extra_requirements=None,
                    extra_packages=None,
                    extra_dirs=None):
  """Generates a Dockerfile for building an image.

  It builds on a specified base image to create a container that:
  - installs any dependency specified in a requirements.txt or a setup.py file,
  and any specified dependency packages existing locally or found from PyPI
  - copies all source needed by the main module, and potentially injects an
  entrypoint that, on run, will run that main module

  Args:
    base_image: (str) ID or name of the base image to initialize the build
      stage.
    main_package: (Package) Represents the main application to execute.
    container_workdir: (str) Working directory in the container.
    container_home: (str) $HOME directory in the container.
    requirements_path: (str) Rath of a requirements.txt file.
    setup_path: (str) Path of a setup.py file
    extra_requirements: (List[str]) Required dependencies to install from PyPI.
    extra_packages: (List[str]) User custom dependency packages to install.
    extra_dirs: (List[str]) Directories other than the work_dir required to be
      in the container.

  Returns:
    A string that represents the content of a Dockerfile.
  """
  is_training_prebuilt_image_base = _IsVertexTrainingPrebuiltImage(base_image)

  dockerfile = textwrap.dedent("""
      FROM {base_image}
      # The directory is created by root. This sets permissions so that any user can
      # access the folder.
      RUN mkdir -m 777 -p {workdir} {container_home}
      WORKDIR {workdir}
      ENV HOME={container_home}

      # Keeps Python from generating .pyc files in the container
      ENV PYTHONDONTWRITEBYTECODE=1
      """.format(
          base_image=base_image,
          workdir=shlex_quote(container_workdir),
          container_home=shlex_quote(container_home)))

  dockerfile += _SitecustomizeRemovalEntry(is_training_prebuilt_image_base)

  dockerfile += _DependencyEntries(
      is_training_prebuilt_image_base,
      requirements_path=requirements_path,
      setup_path=setup_path,
      extra_requirements=extra_requirements,
      extra_packages=extra_packages,
      extra_dirs=extra_dirs)

  dockerfile += _PreparePackageEntry(main_package)
  dockerfile += _GenerateEntrypoint(main_package,
                                    is_training_prebuilt_image_base)

  return dockerfile


def BuildImage(base_image,
               host_workdir,
               main_script,
               output_image_name,
               python_module=None,
               requirements=None,
               extra_packages=None,
               container_workdir=None,
               container_home=None,
               no_cache=True,
               **kwargs):
  """Builds a Docker image.

  Generates a Dockerfile and passes it to `docker build` via stdin.
  All output from the `docker build` process prints to stdout.

  Args:
    base_image: (str) ID or name of the base image to initialize the build
      stage.
    host_workdir: (str) A path indicating where all the required sources
      locates.
    main_script: (str) A string that identifies the executable script under the
      working directory.
    output_image_name: (str) Name of the built image.
    python_module: (str) Represents the executable main_script in form of a
      python module, if applicable.
    requirements: (List[str]) Required dependencies to install from PyPI.
    extra_packages: (List[str]) User custom dependency packages to install.
    container_workdir: (str) Working directory in the container.
    container_home: (str) the $HOME directory in the container.
    no_cache: (bool) Do not use cache when building the image.
    **kwargs: Other arguments to pass to underlying method that generates the
      Dockerfile.

  Returns:
    A Image class that contains info of the built image.

  Raises:
    DockerError: An error occurred when executing `docker build`
  """

  tag_options = ["-t", output_image_name]

  cache_args = ["--no-cache"] if no_cache else []
  command = ["docker", "build"
            ] + cache_args + tag_options + ["--rm", "-f-", host_workdir]

  has_setup_py = os.path.isfile(os.path.join(host_workdir, _DEFAULT_SETUP_PATH))
  setup_path = _DEFAULT_SETUP_PATH if has_setup_py else None

  has_requirements_txt = os.path.isfile(
      os.path.join(host_workdir, _DEFAULT_REQUIREMENTS_PATH))
  requirements_path = _DEFAULT_REQUIREMENTS_PATH if has_requirements_txt else None

  home_dir = container_home or _DEFAULT_HOME
  work_dir = container_workdir or _DEFAULT_WORKDIR

  # The package will be used in Docker, thus norm it to POSIX path format.
  main_package = utils.Package(
      script=main_script.replace(os.sep, posixpath.sep),
      package_path=host_workdir.replace(os.sep, posixpath.sep),
      python_module=python_module)

  dockerfile = _MakeDockerfile(
      base_image,
      main_package=main_package,
      container_home=home_dir,
      container_workdir=work_dir,
      requirements_path=requirements_path,
      setup_path=setup_path,
      extra_requirements=requirements,
      extra_packages=extra_packages,
      **kwargs)

  joined_command = " ".join(command)
  log.info("Running command: {}".format(joined_command))

  return_code = local_util.ExecuteCommand(command, input_str=dockerfile)
  if return_code == 0:
    return utils.Image(output_image_name, home_dir, work_dir)
  else:
    error_msg = textwrap.dedent("""
        Docker failed with error code {code}.
        Command: {cmd}
        """.format(code=return_code, cmd=joined_command))
    raise errors.DockerError(error_msg, command, return_code)
