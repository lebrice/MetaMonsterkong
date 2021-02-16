from setuptools import setup, find_packages
packages = find_packages()

setup(name='meta_monsterkong',
	version='0.0.1',
    description="meta_monsterkong: samples a new map uniformly at random from a directory of generated maps",
	install_requires=[
     'gym',
     'pygame',
    # NOTE: Adding these repos as requirements, and if modifications were made (for
    # instance with gym-ple, it seems like we're using an older version of their repo)
    # then we add a fork of that repo as a dependency.
     "opencv-python",
     "matplotlib",
     'gym_ple @ git+https://github.com/lebrice/gym-ple.git#egg=gym_ple',
     'ple @ git+https://github.com/ntasfi/PyGame-Learning-Environment.git#egg=ple',
    # I don't think this is really necessary if we just want to create the environment:
    # The only reason that I can see is the option of vectoring those environments,
    # which has since been added directly to gym as `gym.vector`. 
    # However if want to retain the ability to render these vectorized envs, then we'd
    # need to use a fork of gym that adds it.
    #  'baselines @ git+https://github.com/openai/baselines.git',
    ],
    packages = packages,
    include_package_data = True,
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        # And include any *.msg files found in the "hello" package, too:
        "meta_monsterkong": ["assets/*", "firsttry/*", "maps20x20/*", "maps20x20_test/*"],
    }
)
