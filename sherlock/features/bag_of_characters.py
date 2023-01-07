import statistics as statistics
from sherlock.features.stats_helper import compute_stats
from collections import OrderedDict
from sherlock.features.helpers import CHARACTERS_TO_CHECK
from sherlock.global_state import is_first
from datetime import datetime

character_mapping = {
 '!':"exclamation",
 '"':"double_quotes",
 '#':"hashtag",
 '$':"dollar",
 '%':"percentage",
 '&':"ampersand",
 "'":"single_quotes",
 '(':"open_round_bracket",
 ')':"close_round_bracket",
 '*':"asterisk",
 '+':"plus",
 ',':"comma",
 '-':"minus",
 '.':"full_stop",
 '/':"forward_slash",
 ':':"colon",
 ';':"semi_colon",
 '<':"less_than",
 '=':"equal_to",
 '>':"greater_than",
 '?':"question_mark",
 '@':"at_the_rate",
 '[':"open_square_bracket",
 ']':"close_square_bracket",
 '_':"underscore",
 '`':"escape_character",
 '{':"open_curly_bracket",
 '|':"vertical_bar",
 '}':"close_curly_bracket",
 '~':"tilde",
 ' ':"space",
 '\x0c':"form_feed",
 '\\':"backward_slash",
 '^':"xor_operator"
}

# Input: a single column in the form of Python list
# Output: ordered dictionary holding bag of character features
def extract_bag_of_characters_features(col_values: list, features: OrderedDict):
    # Create a set of unique chars from the string vectors to quickly test whether to perform expensive
    # processing for any given char
    char_set = set(''.join(col_values))
    
    start_time = datetime.now()
    print('Bag of characters started:',start_time)

    for c in CHARACTERS_TO_CHECK:
        
        if c in character_mapping.keys():
            value_feature_name = f'n_[{character_mapping[c]}]'
        else:
            value_feature_name = f'n_[{c}]'

        if c in char_set:
            counts = [s.count(c) for s in col_values]

            has_any = any(counts)
        else:
            has_any = False

        if has_any:
            _any = 1
            _all = 1 if all(counts) else 0
            _mean, _variance, _skew, _kurtosis, _min, _max, _sum = compute_stats(counts)
            _median = statistics.median(counts)

            if is_first():
                # the first output needs fully expanded keys (to drive CSV header)
                features[value_feature_name + '-agg-any'] = _any
                features[value_feature_name + '-agg-all'] = _all
                features[value_feature_name + '-agg-mean'] = _mean
                features[value_feature_name + '-agg-var'] = _variance
                features[value_feature_name + '-agg-min'] = _min
                features[value_feature_name + '-agg-max'] = _max
                features[value_feature_name + '-agg-median'] = _median
                features[value_feature_name + '-agg-sum'] = _sum
                features[value_feature_name + '-agg-kurtosis'] = _kurtosis
                features[value_feature_name + '-agg-skewness'] = _skew
            else:
                # subsequent lines only care about values, so we can pre-render a block of CSV. This
                # cuts overhead of storing granular values in the features dictionary
                features[value_feature_name + '-pre-rendered'] = \
                    f'{_any},{_all},{_mean},{_variance},{_min},{_max},{_median},{_sum},{_kurtosis},{_skew}'
        else:
            if is_first():
                # the first output needs fully expanded keys (to drive CSV header)
                features[value_feature_name + '-agg-any'] = 0
                features[value_feature_name + '-agg-all'] = 0
                features[value_feature_name + '-agg-mean'] = 0
                features[value_feature_name + '-agg-var'] = 0
                features[value_feature_name + '-agg-min'] = 0
                features[value_feature_name + '-agg-max'] = 0
                features[value_feature_name + '-agg-median'] = 0
                features[value_feature_name + '-agg-sum'] = 0
                features[value_feature_name + '-agg-kurtosis'] = -3
                features[value_feature_name + '-agg-skewness'] = 0
            else:
                # assign pre-rendered defaults
                features[value_feature_name + '-pre-rendered'] = '0,0,0,0,0,0,0,0,-3,0'

    end_time = datetime.now()
    print('Bag of characters completed:', end_time)
    print('Total time taken for bag of characters:',end_time-start_time)
