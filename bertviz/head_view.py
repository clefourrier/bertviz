import json
from IPython.core.display import display, HTML, Javascript
import os
from .util import format_special_chars, format_attention


def head_view(attention, tokens_in, tokens_out, sentence_b_start = None, prettify_tokens=True):
    """Render head view

        Args:
            attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
            tokens: list of tokens
            sentence_b_index: index of first wordpiece in sentence B if input text is sentence pair (optional)
            prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
    """
    vis_html = """
          <span style="user-select:none">
            Layer: <select id="layer"></select>
          </span>
          <div id='vis'></div> 
        """

    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'head_view.js')).read()

    attn = format_attention(attention)
    attn_data = {
        'all': {
            'attn': attn.tolist(),
            'left_text': tokens_in,
            'right_text': tokens_out
        }
    }
    
    params = {
        'attention': attn_data,
        'default_filter': "all"
    }

    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))