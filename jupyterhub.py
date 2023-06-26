#------------------------------------------------------------------------------
# Configuration file for jupyterhub.
#------------------------------------------------------------------------------

# set of users who can administer the Hub itself
c.Authenticator.admin_users = {'kamms'}
c.Authenticator.whitelist = {'kamms'}

# set the Custom logo
#c.JupyterHub.logo_file = '/proteogyver/assets/logo_small.jpg'

## The public facing port of jupyterhub
c.JupyterHub.port = 8090
c.Spawner.notebook_dir = '~/notebooks'
c.Spawner.debug = True

# Change to lab interface
c.Spawner.default_url = '/lab'

# Disable terminal to make it more difficult for users to screw up.
# THIS DOES NOT MITIGATE ANY SECURITY RISKS.
# Users can still cause major mayhem, if they try hard enough.
c.NotebookApp.terminals_enabled = False

# System users:
c.LocalAuthenticator.create_system_users = True

## The command to use for creating users as a list of strings
c.Authenticator.add_user_cmd = ['adduser', '--force-badname', '-q', '--gecos', '""', '--disabled-password']

#Use Google Authenticator
# from oauthenticator.google import GoogleOAuthenticator
# c.JupyterHub.authenticator_class = GoogleOAuthenticator
# c.GoogleOAuthenticator.oauth_callback_url = 'http://example.com/hub/oauth_callback'
# c.GoogleOAuthenticator.client_id = '635823090211-nhef5sl5sqdbq469k4t0l5d14ur7jc8j.apps.googleusercontent.com'
# c.GoogleOAuthenticator.client_secret = 'HA0PdjijSSVog4FUd6nbG9bT'