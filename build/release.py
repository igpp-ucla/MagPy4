import argparse
import sys
import requests
import json

owner = 'igpp-ucla'
repository = 'MagPy4'
def setup_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('token', metavar='T', type=str, nargs=1,
                        help='Personal access token')
    parser.add_argument('--zip', type=str, default=None,
                        required=False,
                        help='Bundled zip file of executable')
    parser.add_argument('--upload-only', action='store_true',
                        help='Skip release creation and only upload zip')
    return parser

def post_request(url, data, token):
    headers = {
        'Accept' : 'application/vnd.github.v3+json',
        'Authorization': f'token {token}'
    }
    response = requests.post(url, json=data, headers=headers)
    response = response.json()
    return response

def main():
    # Parse arguments
    parser = setup_parser()
    result = parser.parse_args(sys.argv[1:])

    # Get token and a zip file name if specified
    token = result.token[0]
    zip_path = result.zip
    upload_only = result.upload_only

    # Read in version
    with open('version.txt', 'r') as fd:
        version = fd.readline().strip('\n')

    # Create new release request or get latest release info
    url = f'https://api.github.com/repos/{owner}/{repository}/releases'
    data = {
        'tag_name' : f'{version}',
        'name' : f'MagPy {version}',
    }
    if not upload_only:
        response = post_request(url, data, token)
    else:
        url = f'{url}/latest'
        headers = {
            'Accept' : 'application/vnd.github.v3+json',
            'Authorization': f'token {token}'
        }
        response = requests.get(url, headers=headers)
        response = response.json()

    # Raise exception if request failed
    if 'url' not in response:
        raise Exception('Release creation/info failed', response)

    # Upload asset if given
    if 'upload_url' in response and zip_path:
        # Read in data
        with open(zip_path, 'rb') as fd:
            zip_data = fd.read()

        # Set up url, data, and headers
        url = response['upload_url'].split('{')[0]
        header = {'Content-Type' : 'application\zip',
                'Accept' : 'application/vnd.github.v3+json',
                'Authorization': f'token {token}'
        }
        params = {'name' : f'MagPy_installer.zip'}

        # Send POST request
        response = requests.post(url, data=zip_data, headers=header, params=params)
        response = response.json()

        if 'url' not in response:
            raise Exception('Asset upload failed', response)

main()