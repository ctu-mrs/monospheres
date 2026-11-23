from glob import glob

from setuptools import setup

package_name = 'monospheres'

# Node scripts are installed as-is into lib/<package> so that `ros2 run monospheres <script>`
# finds them. Their `#!/usr/bin/env python3` shebang picks up the python-env venv when it is
# activated (see create_python_env.sh), which provides the pip-only dependencies.
node_scripts = [
    'scripts/ov_extension.py',
    'scripts/pointcloud_transformer.py',
    'scripts/random_octomap_explorer.py',
    'scripts/goforward5m.sh',
    'scripts/waitForVio.sh',
]

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('lib/' + package_name, node_scripts),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tom',
    maintainer_email='anonymous@todo.todo',
    description='MonoSpheres monocular UAV exploration',
    license='MIT',
)
