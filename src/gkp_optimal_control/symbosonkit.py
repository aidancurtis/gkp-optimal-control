import sympy as sp
from sympy import Add, Mul, Pow, cos, sin, cosh, sinh, exp, factorial, latex, Symbol
from sympy.physics.quantum import Dagger, Commutator, Operator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.operatorordering import normal_ordered_form
from typing import Union

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _normal(expr):
    """Normal–order *and* separate commutative factors."""
    if expr.is_Add:
        return sum(_normal(arg) for arg in expr.args)
    if expr.is_Mul:
        comm, noncomm = sp.Mul(*[f for f in expr.args if f.is_commutative]), \
                        sp.Mul(*[f for f in expr.args if not f.is_commutative], evaluate=False)
        if noncomm is sp.S.One:
            return comm
        return comm * normal_ordered_form(noncomm.expand(), independent=True)
    return expr


def _ad(A, X):
    """adjoint action ad_A(X) = [A, X] with linearity in A."""
    if A.is_Add:
        return _normal(sum(_ad(term, X) for term in A.args))
    return _normal(Commutator(A, X).doit())


def _is_antihermitian(op):
    return sp.simplify(Dagger(op) + op) == 0

def _extract_operators(e):
    """
    Extract all non-commutative operators from the expression e.
    
    Parameters
    ----------
    e : sympy expression
        The expression from which to extract operators.
        
    Returns
    -------
    list
        List of non-commutative operators found in the expression.
        
    Notes
    -----
    This function identifies all instances of sympy.physics.quantum.Operator or its subclasses
    within the expression and returns them as a list.
    
    Examples
    --------
    >>> from sympy import symbols, Add
    >>> from sympy.physics.quantum import Operator
    >>> x, y = symbols('x y')
    >>> A = Operator('A')
    >>> B = Operator('B')
    >>> expr = Add(x, A, y*B)
    >>> _extract_operators(expr)
    [A, B]
    """
    
    return [op for op in e.atoms(Operator) if not op.is_commutative]

# ---------------------------------------------------------------------
# Operator Function
# ---------------------------------------------------------------------
def _replace_op_func(e, variable):

    if isinstance(e, Operator):
        return OperatorFunction(e, variable)
    
    if e.is_Number:
        return e
    
    if isinstance(e, Pow):
        return Pow(_replace_op_func(e.base, variable), e.exp)
    
    new_args = [_replace_op_func(arg, variable) for arg in e.args]
    
    if isinstance(e, Add):
        return Add(*new_args)
    
    elif isinstance(e, Mul):
        return Mul(*new_args)
    
    else:
        return e
        
class OperatorFunction(Operator):
    """TODO: This is not implemented yet."""

    @property
    def operator(self):
        return self.args[0]

    @property
    def variable(self):
        return self.args[1]

    @property
    def free_symbols(self):
        return self.operator.free_symbols.union(self.variable.free_symbols)

    @classmethod
    def default_args(self):
        return (Operator("a"), Symbol("t"))

    def __call__(self, value):
        return OperatorFunction(self.operator, value)
    
    def __new__(cls, *args, **hints):
        if not len(args) in [2]:
            raise ValueError('2 parameters expected, got %s' % str(args))

        return Operator.__new__(cls, *args)



    def __mul__(self, other):

        if (isinstance(other, OperatorFunction) and
                str(self.variable) == str(other.variable)):                
            factors = (self.operator * other.operator).expand()

            factors_t = _replace_op_func(factors, self.variable)
            
            return factors_t
            
                    
        return Mul(self, other)


    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            x = self.operator._eval_power(e)
            if x:
                if isinstance(x, Operator):
                    return OperatorFunction(x, self.variable)
                else:
                    return x
            else:
                return None

    def _eval_commutator_OperatorFunction(self, other, **hints):
        from sympy.physics.quantum.commutator import Commutator

        if self.operator.args[0] == other.operator.args[0]:
            if str(self.variable) == str(other.variable):
                return Commutator(self.operator, other.operator).doit()

        return None

    def _eval_adjoint(self):
        obj = self.operator._eval_adjoint()
        if obj is None:
            obj = Dagger(self.operator)
        return OperatorFunction(obj, self.variable)
    
    def _print_contents_latex(self, printer, *args):
        return r'{{%s}(%s)}' % (latex(self.operator), latex(self.variable))


# ---------------------------------------------------------------------
# BCH Similarity Transform
# ---------------------------------------------------------------------
def similarity_transform(A, B, *, max_order=6, use_bch_only=False):
    '''
    Compute the similarity transformation e^A B e^(-A).
    
    This function calculates the similarity transformation of operator B with respect
    to the generator A. It uses several optimization strategies to avoid expensive
    computations when special cases are detected.
    
    Parameters
    ----------
    A : sympy.Matrix or sympy expression
        The generator of the transformation. Must be anti-Hermitian.
    B : sympy.Matrix or sympy expression
        The operator to transform.
    max_order : int, optional
        Maximum order in the truncated Baker-Campbell-Hausdorff expansion, by default 6.
        Only used if no special pattern is recognized.
    use_bch_only : bool, optional
        If True, skips all special case optimizations and uses BCH expansion directly, by default False.
        
    Returns
    -------
    sympy expression
        The transformed operator e^A B e^(-A).
        
    Notes
    -----
    The function recognizes and optimizes the following special cases:
    1. If [A,B] = 0, returns B (commuting operators)
    2. If [A,B] = c⋅𝟙 (scalar), returns B + c
    3. If [A,B] = k⋅B, returns e^k⋅B
    4. If operators form a 2D algebra like SU(2) or SU(1,1), uses trigonometric or hyperbolic forms
    5. Otherwise falls back to truncated BCH expansion up to max_order
    
    If A is nilpotent, the BCH expansion will terminate early and be exact.
    
    Raises
    ------
    ValueError
        If the generator A is not anti-Hermitian.
    
    Examples
    --------
    >>> import sympy as sp
    >>> x, y = sp.symbols('x y', real=True)
    >>> A = sp.Matrix([[0, -x], [x, 0]])  # anti-Hermitian
    >>> B = sp.Matrix([[1, 0], [0, -1]])
    >>> similarity_transform(A, B)
    Matrix([[cos(2*x), sin(2*x)], [sin(2*x), -cos(2*x)]])
    '''
    
    if not _is_antihermitian(A):
        raise ValueError("Generator A must be anti‑Hermitian.")

    # If use_bch_only is True, directly use BCH expansion
    if use_bch_only:
        result, term = _normal(B), _normal(B)
        for n in range(1, max_order + 1):
            term = _ad(A, term)
            if term == 0:                         # nilpotent ⇒ stop
                break
            result += term / factorial(n)
        return _normal(result)

    comm1 = _ad(A, B)
    if comm1 == 0:
        return _normal(B)

    # --- scalar shift -------------------------------------------------
    if comm1.is_commutative:
        return _normal(B + comm1)

    # --- pure scaling  [A,B]=kB --------------------------------------
    k = sp.Wild('k', commutative=True)
    m = comm1.match(k * B)
    if m:
        k_val = _normal(m[k])
        return _normal(exp(k_val) * B)

    # --- two‑dimensional Lie algebra ---------------------------------
    Op = sp.Wild('Op', commutative=False)
    m1 = comm1.match(k * Op)
    if m1:
        k_val, C = _normal(m1[k]), _normal(m1[Op])
        comm2 = _ad(A, C)

        m2 = comm2.match(sp.Wild('q', commutative=True) * B)
        if m2:
            q_val = _normal(m2[sp.Wild('q')])
            if sp.simplify(q_val + k_val) == 0:
                # SU(2): rotation
                return _normal(cos(k_val) * B + sin(k_val) / k_val * comm1)
            if sp.simplify(q_val - k_val) == 0:
                # SU(1,1): squeezing
                return _normal(cosh(k_val) * B - sinh(k_val) / k_val * comm1)

    # --- fallback: truncated BCH -------------------------------------
    result, term = _normal(B), _normal(B)
    for n in range(1, max_order + 1):
        term = _ad(A, term)
        if term == 0:                         # nilpotent ⇒ stop
            break
        result += term / factorial(n)
    return _normal(result)

# ---------------------------------------------------------------------
# Hamiltonian transformation
# ---------------------------------------------------------------------
def hamiltonian_transformation(H, U: Operator, *, max_order=6, use_bch_only=False, expand=False):
    """
    Transform a Hamiltonian H using the similarity transformation with respect to operator U.
    Transforms the Hamiltonian H by applying the transformation U H Dagger(U) + i d(U)/dt Dagger(U).
    
    Parameters
    ----------
    H : sympy expression
        The Hamiltonian to transform.
    U : Operator
        Single unitary operator to transform the Hamiltonian with.
    max_order : int, optional
        Maximum order in the truncated Baker-Campbell-Hausdorff expansion, by default 6.
    use_bch_only : bool, optional
        If True, skips all special case optimizations and uses BCH expansion directly, by default False.
    expand : bool, optional
        If True, expands and normal orders the resulting expression, by default False.
        
    Returns
    -------
    sympy expression
        The transformed Hamiltonian after applying the similarity transformation.
    """
    A = U.exp
    
    if not _is_antihermitian(A):
        raise ValueError("Operator U must be unitary.")

    ops = _extract_operators(H)

    # Handle +i d(e^U)/dt Dagger(e^U) term
    time_symbols = [s for s in A.free_symbols if s.is_real and s.name == 't']
    Ht = sp.S.Zero

    if time_symbols:
        t = time_symbols[0]
        dA_dt = sp.diff(A, t)
        Ht += sp.I * dA_dt * exp(A) * Dagger(U)

    if not ops:
        # If no operators, just return the expression
        return H + Ht

    # Populate ops with the Dagger's as well
    ops_extended = set()
    for op in ops:
        ops_extended.add(op)
        if len(op.args) == 2 and op.args[1] == 0: # isinstance(op, Dagger)
            ops_extended.add(op.args[0])  # Add non-dagger version
        else:
            ops_extended.add(Dagger(op))   # Add dagger version

    ops = list(ops_extended)

    # Use dummy operators to prevent cascading substitutions
    dummy_to_transformed = {}
    original_to_dummy = {}

    for i, op in enumerate(ops):
        dummy_op = Symbol(f'dummy_op_{i}', commutative=False)
        original_to_dummy[op] = dummy_op
        dummy_to_transformed[dummy_op] = similarity_transform(A, op, max_order=max_order, use_bch_only=use_bch_only)

    # Two-step substitution to avoid cascading
    H = H.subs(original_to_dummy).subs(dummy_to_transformed)

    if expand:
        return _normal(H + Ht)
    else:
        return H + Ht
    

# -------------------------------------------------------------------------
# Drop terms utils
# -------------------------------------------------------------------------
def drop_terms_containing(e, e_drops: list, deep=False):
    """
    Drop terms in expression 'e' that contain any factor listed in 'e_drops'.
    
    Parameters
    ----------
    e : sympy.Expr
        The expression to filter. If it's an instance of sympy.Add, the function will
        analyze each term and selectively keep terms that don't contain any factor in e_drops.
    e_drops : list
        List of factors to check for. Each element can be a sympy expression.
        If an element is a sympy.Mul, all its factors must be present in a term to drop it.
    deep : bool, optional
        If True, the function will recursively check nested expressions for terms to drop.
        If False, it will only check the top-level terms of the expression. Default is False.
    
    Returns
    -------
    sympy.Expr
        The filtered expression with specified terms removed.
    
    Notes
    -----
    - For non-Add expressions, the function returns the input unchanged.
    - For an Add expression, the function checks each term against the e_drops list.
    - If a term contains any single factor from e_drops, it's removed.
    - If an element in e_drops is a product (Mul), all its factors must be present in a term to remove it.
    """
    
    if isinstance(e, Add):
        new_args = []

        for term in e.args:

            keep = True
            for e_drop in e_drops:
                if any([(e_drop - arg).expand() == 0 for arg in term.args]):
                    keep = False

                if isinstance(e_drop, Mul):
                    if all([(any([(f - arg).expand() == 0 for arg in term.args])) for f in e_drop.args]):
                        keep = False
                        
            if deep and term.is_Add:
                # Recursively drop terms in nested Add expressions
                term = drop_terms_containing(term, e_drops, deep=True)
            elif deep and term.is_Mul:
                # Recursively drop terms in nested Mul expressions
                term = drop_terms_containing(term, e_drops, deep=True)

            if keep:
                new_args.append(term)
        e = Add(*new_args)
    elif isinstance(e, Mul):
        # For Mul, we check if any factor in e_drops is present in the expression
        for e_drop in e_drops:
            if any([(e_drop - arg).expand() == 0 for arg in e.args]):
                return sp.S.Zero
        if deep:
            # If deep is True, recursively check each factor
            new_args = [drop_terms_containing(arg, e_drops, deep=True) for arg in e.args]
            e = Mul(*new_args, evaluate=False)
            
    elif any([(e_drop - e).expand() == 0 for e_drop in e_drops]):
        # If the expression itself is in e_drops, return zero
        return sp.S.Zero
    # Return the modified expression
    return e

def drop_terms_matching(expr, patterns, deep=False):
    """
    Remove every additive term that *matches* at least one pattern.

    Parameters
    ----------
    expr : sympy.Expr
        Typically an expanded, normal‑ordered Hamiltonian or equation.
    patterns : sympy.Expr | list[Expr]
        One or several patterns built with Wild symbols, e.g.

            k = Wild('k', exclude=[0])     # any non‑zero integer/expr
            fast  = exp(I*k*t)             # ±ω t harmonic
            beats = exp(I*k*t)*Dagger(a)   # dressed version

    deep : bool  (default False)
        Recursively descend into Mul / nested Add pieces.

    Notes
    -----
    * A term is dropped as soon as **any** factor in it matches a pattern.
    * If `deep=True` the function also removes matching factors inside a
      product and then re‑assembles the remaining ones.
    """
    if not isinstance(patterns, (list, tuple)):
        patterns = [patterns]

    # --- helper: does *obj* match one of the patterns? ------------------
    def _matches(obj):
        return any(obj.match(pat) is not None for pat in patterns)

    # --- recursive core -------------------------------------------------
    if expr.is_Add:
        kept = []
        for term in expr.args:
            # optional recursion
            term_new = drop_terms_matching(term, patterns, deep) if deep else term
            if not _matches(term_new):
                kept.append(term_new)
        return Add(*kept)

    if expr.is_Mul and deep:
        factors = [f for f in expr.args if not _matches(f)]
        if not factors:                     # every factor matched → term gone
            return sp.S.Zero
        return Mul(*factors, evaluate=False)

    # atom
    return sp.S.Zero if _matches(expr) else expr

def drop_c_number_terms(e):
    """
    Drop commuting terms from the expression e.
    
    This function filters out any terms that are commutative (c-numbers) from a sympy expression.
    If the expression is an instance of the Add class, it returns a new Add
    instance with only the non-commutative terms. If the expression is not an Add
    instance, it returns the expression unchanged.
    
    Parameters
    ----------
    e : sympy.core.expr.Expr
        The sympy expression from which to drop commuting terms.
        
    Returns
    -------
    sympy.core.expr.Expr
        The expression with commuting terms (including complex numbers) removed.
        
    Examples
    --------
    >>> from sympy import symbols, Add, I
    >>> from sympy.physics.quantum import Operator
    >>> x, y = symbols('x y')
    >>> A, B = Operator('A'), Operator('B')
    >>> expr = x + A + y*I + B
    >>> drop_c_number_terms(expr)
    A + B
    """

    if isinstance(e, Add):
        return Add(*(arg for arg in e.args if not arg.is_commutative))

    return e

# ---------------------------------------------------------------------
# Collect terms utils
# ---------------------------------------------------------------------
'''def collect_terms(e, op: Union[Operator, list]):
    """
    Collect terms in the expression e that contain the operator(s) op.
    
    Parameters:
    -----------
    e : sympy expression
        The expression to collect terms from
    op : Operator or list of Operators
        Single operator or list of operators to collect terms for.
        When using a list, the order matters - operators are processed sequentially.
    
    Returns:
    --------
    sympy expression
        Expression with terms collected for the specified operator(s)
    """
    if isinstance(op, list):
        # Handle multiple operators
        collected_expr = sp.S.Zero
        remaining_terms = e
        for single_op in op:
            operator_coeffs = remaining_terms.coeff(single_op)
            # Keep only commutative elements in operator_coeffs
            if operator_coeffs.is_Add:
                comm_coeffs = Add(*(arg for arg in operator_coeffs.args if arg.is_commutative))
                operator_coeffs = comm_coeffs

            elif not operator_coeffs.is_commutative: # is Mul and not commutative
                operator_coeffs = sp.S.Zero
                
            remaining_terms = (remaining_terms - operator_coeffs * single_op).expand()
            collected_expr += operator_coeffs * single_op

        # Add remaining terms that do not contain any of the operators
        collected_expr += remaining_terms

        return collected_expr
    else:
        # Handle single operator
        operator_coeffs = e.coeff(op)
        # Keep only commutative elements in operator_coeffs
            if operator_coeffs.is_Add:
                comm_coeffs = Add(*(arg for arg in operator_coeffs.args if arg.is_commutative))
                operator_coeffs = comm_coeffs

            elif not operator_coeffs.is_commutative: # is Mul and not commutative
                operator_coeffs = sp.S.Zero

        remaining_terms = (e - operator_coeffs * op).expand()
        e = operator_coeffs * op + remaining_terms
        return e'''


def _split_comm_noncomm(term):
    """
    Return (comm, noncomm) where
      • comm  is a fully evaluated commuting factor,
      • noncomm  is the product of non‑commuting factors.
    """
    if term.is_Add:
        comm = Add(*(arg for arg in term.args if arg.is_commutative))
        noncomm = Mul(*(arg for arg in term.args if not arg.is_commutative), evaluate=False)
        return comm, noncomm

    if term.is_Mul:
        comm = Mul(*(f for f in term.args if f.is_commutative))
        noncomm = Mul(*(f for f in term.args if not f.is_commutative), evaluate=False)
        return comm, noncomm

    # Otherwise, term is a single factor (e.g. a symbol or an operator)
    if term.is_commutative:
        return term, sp.S.One
    else:
        return sp.S.One, term

def _single_collect(expr, op):
    """
    Collect terms proportional to a *single* operator 'op'.
    The expression is assumed to be already normal‑ordered and expanded.
    """
    collected = sp.S.Zero
    remainder = sp.S.Zero

    for term in sp.Add.make_args(expr):
        comm, noncomm = _split_comm_noncomm(term)

        if not op.is_commutative:
            # Does the non‑commutative tail end with 'op'?
            if noncomm != sp.S.One:
                if noncomm == op:
                    # This term matches → collect it
                    collected += comm
                    continue
        else:
            # If 'op' is commutative, we can just check if it is in the term
            coeff = comm.coeff(op)
            if coeff or op == comm:
                # This term matches → collect it
                collected += comm.coeff(op) * noncomm
                continue

        # Otherwise this term does *not* match → shove to remainder
        remainder += term

    return collected * op + remainder


def collect_terms(expr, op):
    """
    Collects additive terms that terminate in 'op' (or in every op in the list).

    Parameters
    ----------
    expr : sympy expression
        Any bosonic or mixed expression.
    op : Operator | expr | list[Operator | expr]
        One operator/expression, or a list processed sequentially.
        Order matters for the list case.

    Returns
    -------
    sympy expression
        Expression where all matching terms have been combined.
    """
    if not isinstance(op, (list, tuple)):
        op_list = [op]
    else:
        op_list = list(op)

    for single_op in op_list:
        # if not isinstance(single_op, Operator):
        #     raise TypeError("collect_terms expects Operator(s) as 'op'.")
        expr = _single_collect(expr, single_op)

    return expr

def _factor_mode(f):
    """Return mode label (string) for  a  or  Dagger(a)."""
    if len(f.args) == 2 and f.args[1] == 0: # isinstance(f, Dagger)
        return str(f.args[0])
    if isinstance(f, sp.Pow):
        base = f.base
        if len(base.args) == 2 and base.args[1] == 0: # isinstance(base, Dagger)
            return str(base.args[0])
        return str(base)
    return str(f)

# --- product‑level re‑ordering --------------------------------------

def reorder_mul(nc, mode_order=[]):
    """Return a new non‑commutative product with factors grouped by mode.

    Parameters
    ----------
    nc : sympy.Mul | 1
        The non‑commutative part of a term.
    mode_order : list[Operator]
        Desired global order of modes.  Modes not in the list keep their
        relative order after the listed ones.
    """
    if nc == 1 or not isinstance(nc, sp.Mul):
        return nc

    factors = list(nc.args)
    # group by mode
    groups = {}
    for f in factors:
        m = _factor_mode(f)
        groups.setdefault(m, []).append(f)

    ordered_modes = [str(mode) for mode in mode_order]
    tail = [m for m in groups.keys() if m not in ordered_modes]
    full_order = ordered_modes + tail

    new_factors = []
    for m in full_order:
        if m in groups:
            new_factors.extend(groups[m])  # keep intra‑mode order
    return sp.Mul(*new_factors, evaluate=False)

# --- additive re‑ordering -------------------------------------------

def _add_key(noncomm):
    """
    Canonical key for additive sorting.

    • field 0  : tuple of mode names *ignoring daggers* – this groups
                 every string together with its h.c. partner.
    • field 1  : 0 if the whole string starts with “Dagger”, else 1  –
                 ensures the †‑string is immediately BEFORE the plain one.
    • field 2  : full string repr (tie‑breaker, keeps powers/products
                 in deterministic order).
    """
    if noncomm == 1:
        return (('<scalar>',), 0, '')

    # grab mode labels in order of appearance (dagger‑blind)
    if isinstance(noncomm, sp.Mul):
        labels = []
        for f in noncomm.args:
            if isinstance(f, sp.Pow):
                m = _factor_mode(f.base)
                labels.append(f'{m}^{f.exp}')
            else:
                labels.append(_factor_mode(f))
        labels = ''.join(labels)
    elif isinstance(noncomm, sp.Pow):
        labels = f'{_factor_mode(noncomm.base)}^{noncomm.exp}'
    else:
        labels = _factor_mode(noncomm)

    dagger_flag = 0 if str(noncomm).startswith('Dagger(') else 1
    return (labels, dagger_flag, str(noncomm))


def reorder_add(expr, mode_order=[]):
    """Return an *Add* with
         1. each term's non‑commutative tail internally re‑ordered via
            `reorder_mul` (same‑mode factors adjacent, obeying *mode_order*)
         2. additive terms sorted so dagger‑pairs appear next to each other.

    Note: **does not modify coefficients**; collect first, reorder later.
    """
    if not isinstance(expr, sp.Add):
        return expr

    new_terms = []
    for t in expr.args:
        comm, nc = _split_comm_noncomm(t)
        nc_new   = reorder_mul(nc, mode_order)
        new_terms.append(comm*nc_new)

    # sort additive terms by canonical key
    new_terms.sort(key=lambda term: _add_key(_split_comm_noncomm(term)[1]))
    return sp.Add(*new_terms, evaluate=False)


# ---------------------------------------------------------------------
# Coefficients utils
# ---------------------------------------------------------------------
def coeff_of(e, op):
    """
    Return the coefficient of the (single) operator or expression 'op' in expression 'e'. 
    
    Note that this function does not reorder nor recollects the expression, it simply 
    extracts the coefficient of 'op' if it exists. This function expects to receive 
    the expression *after* all terms have been collected using `collect_terms`.

    Parameters
    ----------
    e : sympy expression
        The expression from which to extract the coefficient.
    op : Operator or sympy expression
        The operator or expression whose coefficient is to be extracted.

    Returns
    -------
    sympy expression
        The coefficient of 'op' in 'e'. If 'op' is not present, returns 0.
    """
    
    if e.is_Add:
        for term in e.args:
            comm, noncomm = _split_comm_noncomm(term)
            if not op.is_commutative:
                # Check if 'op' is the non-commutative tail of the term
                if noncomm == op:
                    return term.coeff(op)
            elif op in term.args or term == op:
                return term.coeff(op)

    elif e.is_Mul:
        # Check if 'op' is a factor in the product
        if op in e.args or e == op:
            return e.coeff(op)

    return sp.S.Zero  # If 'op' is not found, return zero

# ---------------------------------------------------------------------
# Expand terms utils
# ---------------------------------------------------------------------

def expand_terms(e, ops, doit=False, **hints):
    """
    Expand terms in the expression 'e' that contain the operator 'op'.
    
    Parameters
    ----------
    e : sympy expression
        The expression to expand.
    ops : Operator | sympy expression | list[Operator | sympy expression]
        Single operator/expression or a list of operators/expressions to expand terms for.
    doit : bool, optional
        If True, calls .doit() on the expression after expansion, by default False.
    hints : dict
        Additional keyword arguments for .expand().
        
    Returns
    -------
    sympy expression
        The expression with terms containing 'op' expanded.
        
    Notes
    -----
    - If 'e' is an instance of sympy.Add, it will expand each term that contains 'op'.
    - If 'e' is not an Add instance, it will try to expand recursively.
    - If 'op' is a list, the function will process each operator sequentially.
    
    Examples
    --------
    >>> from sympy import symbols, Add
    >>> from sympy.physics.quantum import Operator
    >>> x, y = symbols('x y')
    >>> A = Operator('A')
    >>> expr = Add(x, A, y*A)
    >>> expand_terms(expr, A)
    Add(x, A + y*A)
    """

    if not isinstance(ops, (list, tuple)):
        ops = [ops]

    if isinstance(e, Add):
        new_args = []
        for term in e.args:
            for op in ops:
                comm, noncomm = _split_comm_noncomm(term)
                if op in term.args or term == op or (not op.is_commutative and noncomm == op):
                    # Expand the term
                    term = term.expand(**hints)
                    if doit:
                        term = term.doit().factor()
            new_args.append(term)
        result = Add(*new_args)

    elif isinstance(e, Mul):
        for op in ops:
            comm, noncomm = _split_comm_noncomm(term)
            if op in term.args or term == op or (not op.is_commutative and noncomm == op):
                # Expand the product
                e = e.expand(**hints)
                if doit:
                    e = e.doit().factor()
        result = e

    else:
        result = e  # If 'e' is not an Add or Mul, return it unchanged

    if doit and not isinstance(e, (Add, Mul)):
        result = result.doit()

    return result



# ---------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # symbols & operators
    a, b = BosonOp('a'), BosonOp('b')
    adag, bdag = Dagger(a), Dagger(b)
    alpha = sp.symbols('alpha', complex=True)
    omega, t, theta, r = sp.symbols('omega t theta r', real=True)

    # 1) Displacement
    Gd = alpha*adag - sp.conjugate(alpha)*a
    assert sp.simplify(similarity_transform(Gd, a) - (a - alpha)) == 0

    # 2) Phase rotation
    Gr = sp.I*omega*t*adag*a
    assert sp.simplify(similarity_transform(Gr, a)
                       - a*sp.exp(-sp.I*omega*t)) == 0

    # 3) Beam‑splitter
    Gbs = theta*(adag*b - bdag*a)
    a_out = similarity_transform(Gbs, a)
    b_out = similarity_transform(Gbs, b)
    assert sp.simplify(a_out - (a*sp.cos(theta) - b*sp.sin(theta))) == 0
    assert sp.simplify(b_out - (b*sp.cos(theta) + a*sp.sin(theta))) == 0

    # 4) Single‑mode squeezing
    Gsq = r/2*(a**2 - adag**2)          # anti‑Hermitian for real r
    target = a*sp.cosh(r) - adag*sp.sinh(r)
    assert sp.simplify(similarity_transform(Gsq, a) - target) == 0

    print("✓  All similarity‑transform tests passed.")

    # Test Hamiltonian transformation
    # Define a Hamiltonian H
    H = omega * adag * a
    # Define an anti-Hermitian operator U
    U = sp.I * omega * t * (adag * a)
    # Perform the transformation
    transformed_H = hamiltonian_transformation(H, U, max_order=6, use_bch_only=False)
    # Check if the transformed Hamiltonian is correct
    assert sp.simplify(transformed_H) == 0