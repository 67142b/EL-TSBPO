import argparse
import json
import sys
from contextlib import ExitStack

from tqdm import tqdm
sys.path.insert(0, '../')

from web_agent_site.utils import DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH
from web_agent_site.engine.engine import load_products

parser = argparse.ArgumentParser(description='Convert WebShop product data to Lucene JSONL documents.')
parser.add_argument('--file-path', default=DEFAULT_FILE_PATH)
parser.add_argument('--attr-path', default=DEFAULT_ATTR_PATH)
parser.add_argument(
    '--num-products',
    type=int,
    default=None,
    help='Limit products before conversion; use -1 for all products.',
)
args = parser.parse_args()
num_products = None if args.num_products == -1 else args.num_products

all_products, *_ = load_products(
    filepath=args.file_path,
    attrpath=args.attr_path,
    num_products=num_products,
)


output_limits = {
    './resources_100/documents.jsonl': 100,
    './resources/documents.jsonl': None,
    './resources_1k/documents.jsonl': 1000,
    './resources_100k/documents.jsonl': 100000,
}

with ExitStack() as stack:
    outputs = {
        path: (stack.enter_context(open(path, 'w')), limit)
        for path, limit in output_limits.items()
    }
    for index, product in enumerate(tqdm(all_products, total=len(all_products))):
        option_texts = []
        for option_name, option_contents in product.get('options', {}).items():
            option_contents_text = ', '.join(option_contents)
            option_texts.append(f'{option_name}: {option_contents_text}')
        option_text = ', and '.join(option_texts)

        document = {
            'id': product['asin'],
            'contents': ' '.join([
                product['Title'],
                product['Description'],
                product['BulletPoints'][0],
                option_text,
            ]).lower(),
            'product': product,
        }
        serialized = json.dumps(document) + '\n'
        for output, limit in outputs.values():
            if limit is None or index < limit:
                output.write(serialized)
