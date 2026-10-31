"""Specific safeguards against false numeric fragments, units and code execution."""
import runtime
from utils import HERE,OUT,write,sha
from fractions import Fraction
from evidence import numeric_value,arithmetic_evidence,format_evidence

def main():
    tests=[]
    def check(name,condition):
        assert condition,name
        tests.append(name)
    check('exact_decimal',numeric_value('0.6/9')==Fraction(1,15))
    check('nested_fraction',numeric_value(r'\frac{\frac{1}{2}}{3}')==Fraction(1,6))
    check('thousands',numeric_value('30,000*0.15')==4500)
    check('percentage',numeric_value(r'20\%*600')==120)
    check('percentage_equality_abstains',arithmetic_evidence(r'$24/48*100=50\%$')[1]['checked_pairs']==0)
    check('unit_after_math_delimiter_abstains',arithmetic_evidence('$5=500$ cents.')[1]['checked_pairs']==0)
    check('reject_malformed_commas',numeric_value('1,2') is None)
    check('no_code_execution',numeric_value("__import__('os').system('echo unsafe')") is None)
    check('reject_variables_not_strip_numeric_suffix',arithmetic_evidence('$4x+2=18$')[1]['checked_pairs']==0)
    check('units_unknown',arithmetic_evidence(r'$2\text{ hours}=120\text{ minutes}$')[1]['checked_pairs']==0)
    check('rounded_decimal_not_flagged',arithmetic_evidence('$0.0667*60=4$')[1]['violations']==0)
    check('large_integer_false_sum',arithmetic_evidence('$5600+392=6000$')[1]['violations']==1)
    check('detect_fraction_contradiction',arithmetic_evidence(r'$\frac{0.6}{9}=\frac{2}{15}$')[1]['violations']==1)
    check('numeric_subchain_supported',arithmetic_evidence('$x=3+5=9$')[1]['violations']==1)
    check('malformed_and_zero_division_unknown',numeric_value('1/0') is None and numeric_value('(2+3') is None)
    check('bounded_power',numeric_value('10^999999') is None)
    check('powers_not_confused_with_markdown',numeric_value('2**3+4**2')==24)
    check('approximate_symbol_unknown',arithmetic_evidence('$1/3\\approx0.33$')[1]['checked_pairs']==0)
    check('plain_markdown_equation',arithmetic_evidence('Total: **3+5+7=16**')[1]['violations']==1)
    check('repeated_span_not_double_counted',arithmetic_evidence('$2+2=5$\n$$2+2=5$$')[1]['checked_pairs']==1)
    check('heading_conflict',format_evidence('#### Job3\nFinal: \\boxed{250}')[0][0]==1)
    check('symbolic_box',format_evidence(r'\boxed{32-d}')[0][1]==1)
    check('terminal_incomplete',format_evidence('31+32+33=')[0][2]==1)
    check('markdown_closing_not_truncation',format_evidence('**#### 5**')[0][2]==0)
    check('correct_numeric_answer_no_format_flags',format_evidence('#### 250')[0]==[0,0,0])
    write(OUT/'extractor_checks.json',{'passed':True,'checks':tests,'source_sha256':sha(HERE/'evidence.py')})
    print(f'{len(tests)} extractor checks passed',flush=True)

if __name__=='__main__':main()
