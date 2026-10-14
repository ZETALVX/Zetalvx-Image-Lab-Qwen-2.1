"""Qwen-specific catalog. Model licenses are separate from the application license."""
POLICY_VERSION='2026-10-01.qwen21'
QWEN_REPO='Qwen/Qwen-Image-2.1'
def catalog():
    return [{'id':'qwen_image_21','name':'Qwen Image 2.1','kind':'repository',
        'source':'https://huggingface.co/Qwen/Qwen-Image-2.1',
        'license':'Read the license of the selected model revision before use.',
        'policy_version':POLICY_VERSION}]
def external_reason(url):return ''
