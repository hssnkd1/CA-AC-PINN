---
title: Adaptive Circular Physics Informed Neural Networks for Multiscale Partial Differential Equations
subtitle: Mathematical analysis and residual guided refinement
lang: en-US
---

## Abstract

Localized solution features can make a fixed neural representation inefficient even when the governing partial differential equation is simple. We develop an adaptive circular physics-informed network in which compact radial features are added through a nested geometric hierarchy and their output coefficients are determined by a weak variational problem. For symmetric diffusion equations, candidate refinement blocks are ranked by a residual Schur-complement score that equals the reduction in squared energy error under exact integration and exact coefficient solution. We also establish an energy-based removal rule, a density result for unrestricted circular dictionaries, and an error decomposition separating approximation, quadrature and algebraic contributions. A functional flux majorant provides an architecture-independent upper bound at the continuous level. Five reproducible numerical cases cover smooth Poisson diffusion, a discontinuous-coefficient interface problem, a forced Allen–Cahn front, a viscous Burgers travelling wave and heterogeneous transient diffusion. At 129 coefficients, the adaptive interface solution attains a relative $L^2$ error of $1.00\times10^{-2}$, compared with $1.31\times10^{-1}$ for a fixed circular Ritz network. At 64 coefficients and a time step of 0.002, the adaptive Allen–Cahn and Burgers errors are $1.31\times10^{-3}$ and $2.42\times10^{-3}$. The results also identify limits: tanh PINNs are more accurate on the smooth test, classical discretizations remain faster in this implementation, and the low-order flux reconstruction produces pessimistic majorants. The method therefore offers an analyzable mechanism for allocating radial features to localized PDE structure, without implying general superiority over established solvers.

**Keywords** compact radial basis networks; variational physics-informed learning; adaptive approximation; elliptic interface problems; energy error; numerical verification

# 1 Introduction

Partial differential equations with localized gradients, narrow interfaces or heterogeneous coefficients require an approximation space that resolves several spatial scales. Uniformly increasing resolution is reliable but may allocate most of the added degrees of freedom to regions where the solution is already adequately represented. Adaptive finite elements address this issue by modifying a mesh. Neural discretizations offer another possibility: modify the representation itself while preserving a differentiable function that can be evaluated away from the training locations.

Neural solutions of differential equations predate modern deep learning [1]. Physics-informed neural networks subsequently established a practical framework in which differential operators and observations enter a common optimization problem [2,3]. Their flexibility is attractive for inverse problems and coupled models. However, flexibility alone does not determine whether a neural approximation is accurate, stable or economical. Training can be affected by imbalanced derivatives and different convergence rates among loss terms [4,5]. A small empirical residual also does not automatically provide a small error in a physically relevant norm.

Several remedies already exist. Gradient-enhanced losses constrain derivatives of the equation residual [6], while residual-adaptive sampling changes where the differential equation is enforced [7]. Adaptive weighting and gradient-based resampling continue to be developed, including recent studies in Computers & Mathematics with Applications [8,9]. Variational formulations instead exploit the weak problem and reduce spatial differentiability requirements [10,11]. These alternatives address related difficulties, but they modify different parts of the numerical method. Changing sample locations, changing loss weights and changing the approximation space should therefore be evaluated separately.

This paper considers structural adaptation of a compact radial network. Each local feature has a center and radius, and the set of active features changes during the solution process. The term circular refers to Euclidean balls in the coordinates used to construct the features. In one spatial dimension these supports are intervals; in the fitted interface experiment they are circular in a transformed coordinate system. The radial feature itself is an established Wendland function, rather than a new activation.

The main methodological contribution is the explicit connection between a geometric refinement decision and a computable variational energy reduction. For a symmetric elliptic problem, the residual of a proposed group of child features is measured after removing the components already represented by the current space. The resulting Schur-complement score has an exact interpretation. This avoids assigning an unproved energy-error meaning to a patchwise strong residual multiplied by a local length scale.

The paper makes four contributions within this scope. It specifies a compact circular Ritz network with nested hierarchical enrichment and exact boundary traces. It derives refinement and removal identities, together with an approximation and quadrature analysis. It incorporates a classical functional majorant in a way that remains valid for overlapping neural features. It then supplies five numerical cases with saved coefficients, baseline implementations, independent discretizations and verification checks. The numerical study is a coefficient-matched CPU study; it does not establish optimal adaptive rates, high-dimensional efficiency or superiority over tuned modern deep architectures.

# 2 Related work and position of the method

## 2.1 Variational neural discretizations

The Deep Ritz method minimizes an energy over a neural trial family [10]. Variational PINNs and their domain-decomposed extensions use weak equations and explicit test spaces [11]. Berrone, Canuto and Pintore showed that quadrature precision and the relationship between trial and test spaces are central to stability and error analysis [12], and derived an a posteriori estimator for a particular variational discretization [13]. Robust variational PINNs connect neural training to stable residual norms through a Petrov–Galerkin construction [14]. These results motivate treating the neural approximation as a numerical discretization with identifiable consistency and stability requirements.

Here, the geometry is fixed during each coefficient solve. The trial family then becomes a finite-dimensional linear space, so its Ritz equations can be solved directly. Structural learning occurs through selection of geometric features. This reduces the nonlinear optimization burden for the elliptic cases, at the cost of constructing and screening a candidate dictionary.

## 2.2 Localized and adaptive radial representations

Finite basis PINNs use overlapping local neural approximations [15]. Physics-informed radial basis networks establish radial features as an effective local representation for PDEs [16]. More recent AB-PINNs introduce trainable subdomains and residual-driven growth [17]. Sparse RBF networks combine feature selection, optimization and pruning within a functional framework [18]. Greedy neural PDE algorithms also provide rigorous links between variational optimization and selection of additional features [19]. Consequently, locality, radial geometry, network growth and pruning are not independent novelty claims of the present work.

The closest distinction is the specific combination of compact support, a retained-parent hierarchy, a block residual score with an exact energy interpretation, and energy-guarded compression. Table 1 summarizes this position. The paper does not claim that this combination establishes a new state of the art; a direct implementation comparison with [17,18] remains necessary for that stronger conclusion.

**Table 1. Relationship to closely related methods.**

| Method | Representation or adaptation | Relationship to this study |
|:---|:---|:---|
| RAR PINNs [7] | Change residual sampling locations | Baseline that changes sampling while retaining a tanh architecture |
| PIRBN [16] | Radial neural representation | Establishes the prior use of radial features for PDEs |
| FBPINNs and AB-PINNs [15,17] | Local networks and adaptive subdomains | Establish prior localized and structurally adaptive neural solvers |
| Sparse RBF networks [18] | Feature selection and sparsity-based compression | Closest radial structural-learning framework |
| Present AC-PINN | Compact hierarchy and residual Schur-complement selection | A specified Ritz realization with explicit energy accounting |

## 2.3 Approximation and error control

Compact radial approximation has a substantial mathematical foundation [20,21], including the interaction between accuracy and conditioning [22]. Local enrichment is also central to the partition-of-unity method [23]. However, a simple sum of compact radial features is not automatically a partition of unity, and its approximation rates cannot be borrowed without checking the required assumptions.

PINN convergence and error analysis likewise depend on the PDE, the norm, the trial class and the way boundary conditions are imposed [24,25,26]. Certified wavelet-based PINNs already provide residual-based quantitative error control [27]. The flux estimate used below is a classical functional estimate [28], included to distinguish a continuous upper bound from the local indicator used to choose features. Adaptive finite element analysis [29] provides an important reference point, but its contraction and optimality results do not transfer automatically to a changing neural dictionary.

# 3 Mathematical problem formulation

## 3.1 Elliptic problem and energy space

Let Ω be a bounded Lipschitz domain in dimension one or two. Let the scalar diffusion coefficient be measurable and satisfy the uniform ellipticity condition

$$
0<a_{\min}\leq a(x)\leq a_{\max}<\infty\quad\text{a.e. in }\Omega.\qquad (1)
$$

We consider the Dirichlet problem

$$
-\nabla\!\cdot(a\nabla u)=f\quad\text{in }\Omega,\qquad u=g_D\quad\text{on }\partial\Omega.\qquad (2)
$$

Assume f belongs to $L^2$(Ω) and that the boundary data admit an $H^1$ lifting ℓ. Set V = $H_0^1$(Ω). For admissible functions, define

$$
A(w,z)=\int_\Omega a\nabla w\cdot\nabla z\,dx,\quad
F(z)=\int_\Omega fz\,dx,\quad
\|z\|_E^2=A(z,z).\qquad (3)
$$

The weak solution belongs to ℓ + V and satisfies $A(u,z)=F(z)$ for every z in V. Coercivity and the Poincaré inequality give existence and uniqueness. The energy norm measures the diffusion-weighted gradient error and remains meaningful when a is discontinuous.

The associated energy functional is

$$
J(v)=\tfrac12 A(v,v)-F(v),\qquad v\in\ell+V.\qquad (4)
$$

All energy identities below concern this symmetric diffusion problem. They are not asserted for the nonsymmetric Burgers operator or for an arbitrary nonconvex reaction energy.

## 3.2 Interface conditions

For a piecewise coefficient and an interface Γ, a solution without an interfacial source satisfies continuity of the trace and normal diffusive flux,

$$
[u]_\Gamma=0,\qquad [a\nabla u\cdot n]_\Gamma=0.\qquad (5)
$$

The weak formulation incorporates these conditions distributionally. A smooth function in physical coordinates generally does not satisfy the flux condition when a jumps. Applying a strong differential operator separately on each side while ignoring that mismatch would therefore define a different problem. The interface experiment uses a known coefficient-dependent coordinate map that enforces the flux condition for every trial function, and uses the same map for all neural baselines.

## 3.3 Time-dependent extensions

The general diffusion–reaction model is

$$
u_t-\nabla\!\cdot(a\nabla u)+g(u)=f\quad\text{in }\Omega\times(0,T].\qquad (6)
$$

The numerical extension uses backward Euler time stepping with spatial circular features. The coefficient vector changes with time; the representation is not a fully trained space–time network. This choice separates temporal discretization from spatial feature selection and permits independent time-step refinement studies. Section 6 identifies the conditions under which a time-step energy is convex.

# 4 Adaptive circular neural architecture

## 4.1 Compact radial features and boundary enforcement

The approximation has a small global component and a set S of active circular features,

$$
u_S(x,t)=\ell(x,t)+B(x)\left[\sum_{k=1}^{G}\alpha_k(t)p_k(x)+
\sum_{j\in S}c_j(t)\psi\!\left(\frac{\|x-z_j\|}{r_j}\right)\right].\qquad (7)
$$

Here the radii are positive, the centers lie in the computational domain, and B vanishes on the Dirichlet boundary. On the unit square $B(x,y)=16x(1-x)y(1-y)$; on the unit interval $B(x)=4x(1-x)$. These simple factors enforce homogeneous correction traces exactly. Nonhomogeneous data are represented by ℓ. Exact boundary construction follows an established approach [1,30]. A general domain would require an appropriate boundary function or a conforming alternative.

The radial profile is the established compact $C^2$ Wendland function [20],

$$
\psi(s)=(1-s)_+^4(4s+1),\qquad s\geq0.\qquad (8)
$$

The weak equations require first derivatives. The $C^2$ profile also permits the strong-form comparison. Its compact support is a substantive change from the globally supported Gaussian feature proposed in a basic radial PINN, although compact radial functions themselves are classical. Analytic derivatives are given in Appendix A and verified numerically.

For the two-dimensional experiments, the global span consists of tensor Legendre products of degrees zero through two, omitting the product with degree two in both coordinates. Thus G = 8. In one dimension, degrees zero through two give G = 3. The global component represents broad variation, while small circular features resolve local structure. No polynomial reproduction property is assumed for the radial component.

## 4.2 Hierarchy and learned geometry

At the coarsest level, centers occupy a 3 × 3 cell-center grid in two dimensions. At level l, the grid has $3\,2^\ell$ centers per coordinate and radius $2/(3\,2^\ell)$. Splitting a parent makes its four dyadic children available and adds all four to the active representation. The parent remains active. The one-dimensional counterpart adds two children. This retained-parent rule makes each enrichment space contain the previous space.

Three refinement levels produce 765 radial candidates in two dimensions and a total candidate space of dimension 773 after including the global functions. The initial space has dimension 17. After 28 accepted splits, the active dimension is 129. Further checkpoints have 193 and 257 coefficients. A fixed comparison network uses the same eight global functions and 121 circles on an 11 × 11 grid, also giving 129 coefficients.

Centers and radii are learned through discrete selection from this hierarchy. They are not continuously optimized by gradient descent in the reported implementation. The architecture is a one-hidden-layer radial network with a fixed boundary multiplier and polynomial enrichment. Calling it physics-informed refers to how its coefficients and geometry are determined from the PDE.

## 4.3 Coefficient training

For a fixed active set, write the homogeneous correction as a linear combination of basis functions $\phi_i$. With the boundary lifting incorporated into the load, coefficient training solves

$$
K_S c=b_S,\qquad (K_S)_{ij}=A(\phi_j,\phi_i),\qquad
(b_S)_i=F(\phi_i)-A(\ell,\phi_i).\qquad (9)
$$

If the basis is linearly independent, the matrix is symmetric positive definite. Cholesky factorization then replaces a nonconvex weight-optimization problem. Numerical assembly uses composite Gaussian quadrature. Small or negative Schur pivots are rejected; the implementation uses a relative pivot threshold of $10^{-12}$. Conditioning is monitored independently of the residual score because a nearly dependent feature can otherwise appear useful through numerical cancellation.

This architecture retains analytic spatial evaluation after training. It also carries a representation cost beyond the output coefficients. At 129 coefficients, 121 circles store 363 center-and-radius scalars in addition to those coefficients. The 129-parameter tanh baseline stores all of its trainable geometry and output weights within its 129 parameters. Equality of trained scalar counts therefore does not imply equality of total storage.

# 5 Residual guided structural adaptation

## 5.1 Residual score for a candidate split

Let $v_S$ be the exact Ritz solution in the current space and let C contain a candidate parent’s child features. Form the block residual and Schur complement

$$
r_C=b_C-K_{CS}K_S^{-1}b_S,\qquad
H_C=K_{CC}-K_{CS}K_S^{-1}K_{SC}.\qquad (10)
$$

The selection score is

$$
\Delta_C=r_C^T H_C^{-1}r_C.\qquad (11)
$$

The Schur complement removes feature components already represented by the active space. Consequently, the score accounts for both the residual and the redundancy of the proposed children. A raw coefficient magnitude or a strong residual at the feature center does not have this property.

Only unsplit active parents with available children are eligible. The algorithm chooses the parent with the largest finite score and adds its complete child block. All scores refer to the currently selected candidate family. A small score can indicate that this finite dictionary is exhausted; it does not certify that the PDE error is small.

## 5.2 Energy guarded pruning and hierarchical merging

After solving the enriched problem, individual circular features can be tested for removal. If c is the optimized coefficient vector, the exact increase in the quadratic energy after removing feature j and reoptimizing the others is

$$
\delta J_j=\frac{c_j^2}{2(K_S^{-1})_{jj}}.\qquad (12)
$$

The implementation repeatedly tests the lowest-cost removable feature and accepts it only while the cumulative energy increase remains within a prescribed budget. Global polynomial functions are protected. The compression study uses budgets equal to one half of τ times the squared energy norm of the uncompressed solution, with τ equal to $10^{-8}$, $10^{-6}$ or $10^{-4}$. This budget controls a change in energy, not the relative error with respect to an unknown exact solution.

The same principle permits a hierarchical merge: retain a parent and remove an optimized group of its descendants, then recompute the coefficients and compare energies. Such a merge is a coarsening operation, not an assertion that the sum of several radial features equals one radial feature. The numerical compression results below test individual pruning. They do not provide an empirical evaluation of merging or continuously optimized replacement centers.

## 5.3 Algorithm

**Algorithm 1. Adaptive circular Ritz solution for symmetric diffusion.**

1. Define the boundary lifting, global functions, compact radial hierarchy and numerical integration rule.
2. Assemble the candidate stiffness matrix and load; initialize the global and coarsest circular features.
3. Solve the current coefficient system and identify eligible unsplit parents.
4. Compute the residual and Schur complement for each proposed child block.
5. Select the admissible block with the largest score and retain the parent while adding the children.
6. Update the factorization, reoptimize all active coefficients and record the energy decrease.
7. Continue to the specified coefficient budget or stop when no stable eligible block remains.
8. If compression is requested, test removals with a cumulative energy-growth budget.
9. Reevaluate the solution on independent integration points and check sensitivity to integration refinement.

An error tolerance can replace the coefficient budget only when a trustworthy error estimate is available. The coarse flux reconstruction used in the present experiments is too pessimistic for an efficient automatic stopping rule.

## 5.4 Computational cost

The reference implementation constructs a full candidate Gram matrix and stores it densely after sparse weighted products have been accumulated. If D is the candidate dimension and M the active dimension, matrix storage is $O(D^2)$, and the incremental orthogonalization factors require $O(DM)$ storage. For single-feature updates, the cumulative algebraic update cost is $O(DM^2)$, apart from candidate-block ranking and selected-system factorization. A direct active solve costs $O(M^3)$. These costs are paid during construction, even when M is much smaller than D.

Basis values and derivatives are evaluated in chunks to limit temporary memory. Compact support creates zero entries for assembly, but the current evaluation loop still visits all candidate features. Likewise, value-only prediction visits all active features and has $O(NM)$ work for N query points. Spatial neighbor indexing and genuinely local sparse assembly could reduce these costs; neither improvement is assumed in the reported timings. The hierarchy also has exponentially many cells as dimension increases, so the one- and two-dimensional experiments support no claim of dimension-independent complexity.

# 6 Mathematical analysis

## 6.1 Variational energy and residual identities

**Proposition 1. Energy identity and best approximation.** Let u solve the weak problem, and let $v_S$ be the exact Ritz solution in an admissible affine finite-dimensional space ℓ + $V_S$. For any admissible v,

$$
2\{J(v)-J(u)\}=\|u-v\|_E^2.\qquad (13)
$$

Moreover, $v_S$ is the energy-orthogonal projection of u onto that affine space and minimizes the energy error there.

*Proof.* Set $e=v-u$, which has zero trace. Expanding J(u+e) and using $A(u,e)=F(e)$ eliminates the linear term and gives the identity. Subtracting the Ritz and weak equations gives A($u-v_S$,z) = 0 for every z in $V_S$. The orthogonal projection property follows by expanding the squared energy distance to any function in the trial space. ∎

Define the residual functional $R_v(z)=F(z)-A(v,z)$, and equip the dual space with the norm induced by the energy norm. The weak equation immediately yields

$$
\|u-v\|_E=\sup_{0\ne z\in V}\frac{|R_v(z)|}{\|z\|_E}.\qquad (14)
$$

This is an exact functional statement. Replacing the supremum by finitely many candidate tests gives only partial information about the full residual.

## 6.2 Exact interpretation of the refinement score

**Proposition 2. Block refinement identity.** Suppose the combined current and child features are linearly independent. Let $v_{\mathrm{new}}$ be the exact Ritz solution after adding C. Then $H_C$ in Eq. (10) is positive definite and

$$
\|u-v_S\|_E^2-\|u-v_{\mathrm{new}}\|_E^2
=2\{J(v_S)-J(v_{\mathrm{new}})\}=\Delta_C.\qquad (15)
$$

*Proof.* Project each proposed child feature onto $V_S$ in the energy inner product and subtract that projection. The resulting child corrections are orthogonal to $V_S$ and have Gram matrix $H_C$. Their residual load is $r_C$. Minimization in this orthogonal complement gives coefficients $H_C^{-1}r_C$ and squared correction norm $r_C^TH_C^{-1}r_C$. Orthogonality yields the difference of squared energy errors, while Proposition 1 gives the difference of energies. ∎

With floating-point quadrature, the same algebraic identity holds for the quadrature-defined positive-definite form. It need not imply monotone decay of the continuous error if integration is insufficient. This distinction is tested directly in Section 8.

**Proposition 3. Removal cost.** For an optimized quadratic coefficient problem with positive-definite matrix $K_S$, Eq. (12) is the exact minimum energy increase caused by the constraint that coefficient j becomes zero.

*Proof.* Let $\delta c$ denote the coefficient change. Since the original gradient vanishes, the energy increase is one half of $\delta c^T K_S\delta c$. Minimize this quantity subject to $\delta c_j=-c_j$. A Lagrange multiplier gives $\delta c=-c_jK_S^{-1}e_j/(K_S^{-1})_{jj}$. Substitution yields Eq. (12). For several simultaneous removals the corresponding principal inverse block gives the analogous constrained quadratic minimum. ∎

These are classical projection and quadratic-optimization identities specialized here to circular structural operations. Their role is to justify the implemented decisions; they are not presented as new general identities in numerical analysis.

## 6.3 Approximation properties and their limits

**Proposition 4. Density of unrestricted circular dictionaries.** On the unit interval or square, let B be the boundary factor above. Allow arbitrary finite sums of translated and scaled copies of ψ, with centers in the interior and positive radii tending to zero. Their products with B are dense in $H_0^1$(Ω).

*Proof.* Choose a smooth compactly supported approximation $u_\varepsilon$ to an arbitrary function in $H_0^1$(Ω). Since B is positive on the support of $u_\varepsilon$, the quotient $q=u_\varepsilon/B$ extends to a smooth compactly supported function. Normalize the radial profile to a nonnegative compactly supported kernel k of unit integral. Standard approximate-identity convergence gives $k_\rho*q\to q$ in $H^1$ as ρ tends to zero. For each fixed radius, Riemann sums for the convolution converge in $H^1$ to a finite linear combination of translated copies of $k_\rho$. Multiplication by B is bounded in $H^1$. Combining the initial smooth approximation, the convolution approximation and the finite-sum approximation proves density. ∎

A conservative quantitative version clarifies what this argument does and does not establish. For a smooth compactly supported q, let h be the maximum cell diameter of a Riemann partition of a fixed interior box, and let 0 < ρ ≤ 1. There is a finite circular sum $s_{\rho,h}$ such that

$$
\|Bq-Bs_{\rho,h}\|_{H^1}
\leq C\left[\rho^2\|q\|_{H^3}
+h\rho^{-d/2-2}\|q\|_{W^{1,\infty}}\right].\qquad (16)
$$

The constant depends on the box, B and the normalized kernel. A derivation is provided in Appendix B. The estimate implies convergence when $h\rho^{-d/2-2}$ tends to zero, but is deliberately conservative. It is not a sharp rate for the implemented hierarchy, whose center spacing and radius are proportional. Proposition 4 concerns the unrestricted family; it does not prove convergence of Algorithm 1 at a fixed maximum depth or an optimal error-versus-feature rate. Establishing a stable multilevel approximation theorem for that particular hierarchy remains an open part of the analysis.

## 6.4 Approximation quadrature and algebraic errors

Let $u_S$ be the exact continuous Ritz solution in a selected space, and let $v_Q$ be the numerical coefficient solution in that same space. Write $A_Q$ and $F_Q$ for the numerical form and load. Define two restricted dual defects,

$$
\begin{aligned}
\delta_Q(v_Q)&=\sup_{0\ne z\in V_S}
\frac{|(F-F_Q)(z)+(A_Q-A)(v_Q,z)|}{\|z\|_E},\\
\delta_{\mathrm{alg}}(v_Q)&=\sup_{0\ne z\in V_S}
\frac{|F_Q(z)-A_Q(v_Q,z)|}{\|z\|_E}.
\end{aligned}\qquad (17)
$$

**Proposition 5. Fixed-space error decomposition.** Under the assumptions above,

$$
\|u-v_Q\|_E^2\leq
\inf_{w\in\ell+V_S}\|u-w\|_E^2+
\{\delta_Q(v_Q)+\delta_{\mathrm{alg}}(v_Q)\}^2.\qquad (18)
$$

*Proof.* For z in $V_S$, subtracting the equations gives $A(u_S-v_Q,z)$ = $(F-F_Q)(z)+(A_Q-A)(v_Q,z)+F_Q(z)-A_Q(v_Q,z)$. Taking the restricted energy dual norm bounds the within-space error. Galerkin orthogonality makes $u-u_S$ orthogonal to $u_S-v_Q$, so the Pythagorean identity and Proposition 1 give Eq. (18). ∎

Numerical refinement of an integration rule is evidence about $\delta_Q$, not an interval enclosure of it. Similarly, a small coefficient-system residual must be interpreted relative to matrix conditioning. This is why the study reports independent error integration and selected-space quadrature checks instead of using the training residual alone.

## 6.5 Continuous functional majorant

**Proposition 6. Flux reliability bound.** Let y belong to $H(\operatorname{div},\Omega)$, and let $C_P$ satisfy the Poincaré inequality on V. For every admissible v,

$$
\|u-v\|_E\leq
\|a^{-1/2}(y-a\nabla v)\|_{L^2}
+\frac{C_P}{\sqrt{a_{\min}}}\|f+\nabla\!\cdot y\|_{L^2}
\equiv\mathcal M(v,y).\qquad (19)
$$

*Proof.* Test the error equation with $e=u-v$ and add y. Integration by parts is valid for y in $H(\operatorname{div},\Omega)$, and the boundary contribution vanishes because e has zero trace. Cauchy–Schwarz bounds the flux defect by its weighted $L^2$ norm times the energy norm of e. The equilibrium term is bounded by $C_P/\sqrt{a_{\min}}$ times the $L^2$ norm of f+div y times the same energy norm. Division proves the result. ∎

This is a standard functional estimate [28]. Its applicability does not depend on Galerkin orthogonality, on a mesh for the trial function, or on the support shape. On the unit square, $C_P=1/(\pi\sqrt{2})$ is admissible. A rectangular lowest-order Raviart–Thomas space provides a conforming recovered flux [31]. The numerical reconstruction minimizes a weighted square form of the two terms and updates their balancing parameter.

Data variation within a flux cell must remain in the equilibrium defect. Because the RT0 divergence is constant on each cell K,

$$
\|f+\nabla\!\cdot y\|_{L^2(K)}^2
=|K|(\bar f_K+\nabla\!\cdot y)^2+\|f-\bar f_K\|_{L^2(K)}^2.\qquad (20)
$$

Removing the last term would underestimate the majorant. In the experiments the complete expression is integrated numerically. Thus the continuous inequality is rigorous, but the reported numerical values are quadrature-evaluated estimates rather than certified bounds. No efficiency estimate or contrast-independent constant is claimed.

## 6.6 Scope of the nonlinear extension

Backward Euler applied to Allen–Cahn has the incremental functional

$$
J_n(v)=\frac{\|v-u^{n-1}\|_{L^2}^2}{2\Delta t}
+\frac{\epsilon^2}{2}\|\nabla v\|_{L^2}^2
+\int_\Omega\left(\tfrac14 v^4-\tfrac12 v^2-s^n v\right)dx.\qquad (21)
$$

Its second variation is bounded below by $(1/\Delta t-1)$ times the squared $L^2$ norm plus $\epsilon^2$ times the squared gradient norm. Hence Δt < 1 makes the time-step functional strictly convex on an affine boundary space. A nonlinear energy decrease is then a possible acceptance safeguard. However, the linear block formula in Eq. (11) is not the exact nonlinear energy decrease. The implemented evolution heuristic uses a mass-plus-diffusion residual score and recomputes the nonlinear solution after enrichment. The Burgers extension is also a residual-guided heuristic, and the elliptic refinement proof is not transferred to it.

# 7 Numerical experiment design

## 7.1 Verification strategy and metrics

Every case has an analytic solution or an explicitly manufactured field. Sources and boundary data are derived from that field. This follows the verification purpose of the method of manufactured solutions [32]. Such tests assess whether the stated equations are solved correctly; they do not validate a physical model against observations. No experimental or clinical data are used.

The principal spatial metrics are relative $L^2$ and relative diffusion energy errors,

$$
E_2=\frac{\|u-u_h\|_{L^2}}{\|u\|_{L^2}},\qquad
E_E=\frac{\|u-u_h\|_E}{\|u\|_E}.\qquad (22)
$$

For the one-dimensional front tests, the gradient error is reported separately; the constant diffusion weight cancels from its relative form. Sample maxima are retained in the raw data but are not called exact $L^\infty$ norms. Two-dimensional circular errors are evaluated on an independent composite rule with 80 subdivisions per coordinate and three Gauss points per subinterval, giving 57,600 points. Elliptic training uses 64 subdivisions and 36,864 points. The selected 129-dimensional space is also refitted with 32 and 96 subdivisions. The one-dimensional front calculations use 256 subintervals with four Gauss points each.

## 7.2 Baselines and computational accounting

The elliptic comparisons include a fixed circular Ritz network, a strong-residual least-squares network on the same fixed circles, and three fully trained tanh variants. Each tanh network has two inputs, one hidden layer with 32 units and a scalar output, totaling 129 trainable parameters. The output is multiplied by the same boundary factor. Standard PINNs use 2,048 scrambled Sobol points. The RAR variant starts with 1,024 points and adds 256 large-residual points after each of two training stages, ending with 1,536. It is a specified RAR implementation, not a reproduction of the RAD or RAR-D algorithms in [7].

The gradient-enhanced baseline starts with 1,024 points and adds a weight of $10^{-3}$ on squared residual-gradient estimates. These gradients are approximated by centered differences with increment $10^{-4}$; differences crossing the material interface or the boundary are excluded. Consequently, this baseline is labeled FD gPINN and is not an automatic-differentiation implementation of [6]. Analytic parameter derivatives are used for optimization in all three variants. L-BFGS-B receives 1,500 iterations in total, divided equally among the RAR stages. Seeds zero, one and two define three independent initializations and point sets. Means and sample standard deviations describe these runs; no significance test is performed.

Independent classical calculations use uniform triangular P1 finite elements for the elliptic cases, central finite differences for the one-dimensional fronts, and a conservative variable-coefficient finite difference operator for transient diffusion. The front references use 511 interior unknowns, and the transient reference uses 6,241. Time integration is backward Euler for both network and finite difference comparisons, reducing a temporal-scheme confound. An adaptive FEM comparison and an extensive hyperparameter search are outside the executed study.

All experiments run in CPU double precision with one BLAS thread. The complete study is executed sequentially. Reported construction times include basis assembly and fitting or optimizer work, and exclude reference error integration, figure production and flux postprocessing. Times are individual-run measurements for deterministic methods and averages over the three neural seeds. They are illustrative machine-specific timings, not repeated performance benchmarks. The trained coefficient budgets are matched, but total stored parameters and elapsed budgets are not. The full candidate construction cost is charged to the adaptive method rather than hidden as preprocessing.

## 7.3 Case 1 smooth Poisson diffusion

On the unit square take a = 1, homogeneous Dirichlet data, and

$$
u(x,y)=\sin(\pi x)\sin(\pi y),\qquad f=2\pi^2u.\qquad (23)
$$

This case tests smooth approximation, derivative implementation and the possibility that a global representation already provides an efficient solution. It is not intended to favor localization.

## 7.4 Case 2 discontinuous diffusion with a localized feature

Let a = 1 for x ≤ 1/2 and a = κ = 10 for x > 1/2. Introduce $S_0=1/2+1/(2\kappa)$, and define the continuous coordinate

$$
\xi(x)=\frac1{S_0}\int_0^x a(s)^{-1}\,ds,\qquad
\xi'(x)=\frac1{S_0a(x)}.\qquad (24)
$$

For any smooth function of (ξ,y), the normal flux $a\partial_xu=S_0^{-1}\partial_\xi u$ is continuous at the interface. The network’s circular supports are defined in (ξ,y), not in physical (x,y).

Set $C_0=(1/8+3/(8\kappa))/S_0$ and $U(x)=\int_0^x(C_0-s)/a(s)\,ds$. Then U(0) = U(1) = 0 and $-(aU\prime)\prime=1$. The manufactured solution adds a localized feature,

$$
u(x,y)=U(x)\sin(\pi y)+0.22 B(\xi,y)
\exp\!\left[-\frac{(\xi-0.46)^2+(y-0.43)^2}{0.07^2}\right].\qquad (25)
$$

The forcing is −a times the piecewise physical Laplacian of this field. The flux condition eliminates any distributional interface source. Training integration cells align with x = 1/2. The map is supplied to all neural methods because the coefficient and interface are known; this is not a learned interface-discovery experiment.

## 7.5 Case 3 forced Allen Cahn moving front

The interval problem is $u_t=\epsilon^2u_{xx}+u-u^3+s$, with ε = 0.02. Inspired by the interface structure of the Allen–Cahn model [33], use the exact front

$$
u(x,t)=\tanh\!\left(\frac{x-0.45-0.2t}{\sqrt2\epsilon}\right),
\qquad s(x,t)=u_t(x,t),\qquad 0\leq t\leq0.1.\qquad (26)
$$

The stationary profile balances diffusion and reaction, so the added source produces the prescribed motion. Boundary and initial values are taken from Eq. (26), with a linear time-dependent boundary lifting. This is a forced verification problem, not an unforced coarsening simulation. The adaptive network grows from six coefficients to a maximum of 64. Up to eight splits are allowed at the first step and up to two at later steps. Newton iterations use a residual tolerance of $10^{-10}$ and damping when required.

## 7.6 Case 4 viscous Burgers travelling wave

For $u_t+uu_x=\nu u_{xx}$, set ν = 0.005 and use

$$
u(x,t)=c-A\tanh\!\left(\frac{A(x-ct-x_0)}{2\nu}\right),
\quad c=0.3,\quad A=0.5,\quad x_0=0.35.\qquad (27)
$$

This travelling wave is consistent with the classical viscous Burgers setting [34]. Exact initial and boundary values are prescribed on the unit interval, and the final time is 0.2. The 64-coefficient budget and refinement schedule match the Allen–Cahn case. The convective term is integrated in its primal weak equation and solved by a nonsymmetric Newton system. A centered finite difference discretization provides an independent calculation at the same time steps.

## 7.7 Case 5 heterogeneous transient diffusion

On the unit square, take $a(x,y)=1+0.5\sin(2\pi x)\sin(2\pi y)$, and define

$$
\begin{aligned}
u(x,y,t)&=e^{-t}u_0(x,y),\\
u_0(x,y)&=\sin(\pi x)\sin(\pi y)+0.3B(x,y)
\exp\!\left[-\frac{(x-0.65)^2+(y-0.35)^2}{0.07^2}\right].
\end{aligned}\qquad (28)
$$

The source is $f=-u-\nabla\cdot(a\nabla u)$. Homogeneous boundary values and the exact initial field are imposed weakly through the first time step. The adaptive geometry is selected in that first mass-plus-diffusion solve and then retained. Since the exact solution has a fixed spatial shape, this case assesses transient coefficient evolution and heterogeneous diffusion rather than moving-feature tracking. Spatial assembly uses 48 subdivisions with three Gauss points, and the final time is 0.1.

# 8 Numerical results

## 8.1 Smooth and interface elliptic accuracy

**Table 2. Smooth Poisson results. Neural optimization rows give mean ± sample standard deviation over three seeds; time is the mean. Other rows are deterministic single runs. N is the fitted coefficient or classical interior-unknown count.**

| Method | N | Relative $L^2$ | Relative energy | Time s |
| :--- | :--- | :--- | :--- | :--- |
| AC-PINN | 129 | 6.92e-05 | 7.27e-04 | 6.079 |
| Fixed circular Ritz | 129 | 1.07e-04 | 1.25e-03 | 1.384 |
| Fixed circular strong | 129 | 1.21e-03 | 3.19e-03 | 2.825 |
| tanh PINN | 129 | 1.72e-05 ± 7.47e-06 | 6.28e-05 ± 2.05e-05 | 3.012 |
| RAR tanh PINN | 129 | 3.38e-05 ± 2.92e-05 | 8.14e-05 ± 4.02e-05 | 2.037 |
| FD gPINN | 129 | 1.43e-05 ± 4.47e-06 | 5.32e-05 ± 1.04e-05 | 7.986 |
| P1 FEM | 225 | 1.08e-02 | 9.79e-02 | 0.002 |
| P1 FEM | 961 | 2.70e-03 | 4.91e-02 | 0.006 |
| P1 FEM | 3969 | 6.76e-04 | 2.45e-02 | 0.027 |


**Table 3. Interface diffusion results. Neural optimization rows give mean ± sample standard deviation over three seeds; time is the mean. Other rows are deterministic single runs. N is the fitted coefficient or classical interior-unknown count.**

| Method | N | Relative $L^2$ | Relative energy | Time s |
| :--- | :--- | :--- | :--- | :--- |
| AC-PINN | 129 | 1.00e-02 | 5.72e-02 | 5.995 |
| Fixed circular Ritz | 129 | 1.31e-01 | 5.03e-01 | 1.292 |
| Fixed circular strong | 129 | 2.66e-01 | 5.66e-01 | 2.676 |
| tanh PINN | 129 | 3.86e-01 ± 2.87e-02 | 6.56e-01 ± 2.05e-02 | 3.600 |
| RAR tanh PINN | 129 | 1.11e+00 ± 4.97e-01 | 8.52e-01 ± 8.40e-02 | 2.384 |
| FD gPINN | 129 | 5.06e-01 ± 3.72e-02 | 7.00e-01 ± 6.37e-02 | 10.682 |
| P1 FEM | 225 | 1.37e-01 | 5.00e-01 | 0.002 |
| P1 FEM | 961 | 6.93e-02 | 3.79e-01 | 0.007 |
| P1 FEM | 3969 | 1.93e-02 | 2.02e-01 | 0.028 |



On the smooth Poisson problem, the adaptive network improves on the fixed circular space, but the tanh networks provide lower errors at the same trained scalar count. The adaptive relative $L^2$ error at 129 coefficients is $6.92\times10^{-5}$, while the mean standard tanh PINN error is $1.72\times10^{-5}$. Thus locality and structural growth are not automatically beneficial for a broadly smooth field. The fixed strong-residual circular solve is less accurate than the corresponding Ritz solve in this example, showing that the training formulation also matters.

The interface problem exhibits a different outcome. With 129 coefficients, adaptation reduces the relative $L^2$ error from 0.131 for fixed circular Ritz approximation to 0.0100, a factor of approximately 13.1. The relative energy error falls from 0.503 to 0.0572. The standard tanh baseline has mean relative $L^2$ error 0.386 under the specified training configuration. RAR and FD gPINN do not repair that baseline here. Their large errors should be interpreted as failures of these particular configurations, rather than evidence that all sampling or gradient-enhanced strategies fail on interface problems.

![Figure 1. Relative $L^2$ and energy errors against active coefficient count for the two elliptic cases. Solid curves show nested adaptive spaces. Fixed-circle results at 129 coefficients are included for comparison.](figures/figure_1_elliptic_convergence.png){width=6.3in}

The convergence curves show substantial early gains in the interface problem, followed by slower improvement at larger budgets. The finite dictionary has a smallest radius of 1/12 in transformed coordinates. It cannot continue refining beyond that level, and retained broad functions can remain in the representation even when much of the residual is localized. Increasing the coefficient count within this finite family is therefore different from allowing arbitrary geometric resolution.

![Figure 2. Exact interface field, adaptive and fixed predictions, and adaptive absolute error at 129 coefficients. The dashed line marks the coefficient interface in physical coordinates.](figures/figure_2_interface_fields.png){width=6.3in}

![Figure 3. Selected circular supports in transformed coordinates at three adaptive budgets. Small supports concentrate around the manufactured local feature; the large-scale component remains represented by retained broad features.](figures/figure_3_selected_supports.png){width=6.3in}

The P1 results provide a useful numerical reference. On successively refined uniform grids, both elliptic errors decrease. On the smooth problem, P1 relative $L^2$ errors fall from $1.08\times10^{-2}$ at 225 unknowns to $6.76\times10^{-4}$ at 3,969 unknowns. On the interface test the 3,969-unknown error is $1.93\times10^{-2}$. The adaptive circular network uses fewer fitted coefficients to attain lower error in these examples, but its candidate assembly is substantially more expensive than the sparse FEM calculations. A coefficient count and a classical degree-of-freedom count are not interchangeable measures of total work.

## 8.2 Computational work

![Figure 4. Relative $L^2$ error against measured construction time. Adaptive checkpoints include the complete candidate assembly cost. Tanh markers use mean errors and times over three seeds. Classical solvers are faster in this implementation.](figures/figure_4_error_work.png){width=6.3in}

The timing results do not support a general acceleration claim. Constructing the adaptive candidate matrix costs more than fitting the fixed radial space, and the sparse classical solvers are fastest in these low-dimensional tests. The advantage demonstrated by structural adaptation is selective improvement of the representation at a fixed coefficient count. It is not an established reduction in wall-clock time, memory, latency or energy consumption. An equal-time comparison with tuned baselines, multiple repeated timing runs and a sparse local implementation would be required to evaluate those stronger claims.

## 8.3 Moving nonlinear fronts

**Table 4. Nonlinear front results at a time step of 0.002. Gradient errors are relative $H^1$ seminorm errors.**

| Case | Method | N | Relative $L^2$ | Relative gradient | Time s |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Allen–Cahn | Adaptive circular | 64 | 1.31e-03 | 2.09e-02 | 0.258 |
| Allen–Cahn | Fixed circular | 64 | 7.16e-03 | 3.75e-01 | 0.158 |
| Allen–Cahn | Central FDM | 511 | 7.71e-04 | 6.90e-03 | 0.041 |
| Burgers | Adaptive circular | 64 | 2.42e-03 | 2.24e-02 | 0.380 |
| Burgers | Fixed circular | 64 | 1.54e-02 | 3.13e-01 | 0.293 |
| Burgers | Central FDM | 511 | 1.25e-03 | 1.38e-02 | 0.094 |



At a time step of 0.002, the adaptive Allen–Cahn $L^2$ error is $1.31\times10^{-3}$, compared with $7.16\times10^{-3}$ for the fixed network. The corresponding Burgers errors are $2.42\times10^{-3}$ and $1.54\times10^{-2}$. The gradient errors show a larger difference because unresolved local derivatives contribute relatively little to a global $L^2$ metric. All recorded Newton solves finish within the prescribed iteration limit.

The independent finite difference solutions are more accurate and faster at the reported resolution. For example, the Burgers FDM error is $1.25\times10^{-3}$ at the same time step. The smaller circular coefficient count is therefore a representation result, not a demonstration of superior overall numerical efficiency.

![Figure 5. Final-time Allen–Cahn and Burgers profiles, together with absolute spatial errors near each moving front, for a time step of 0.002.](figures/figure_5_moving_fronts.png){width=6.3in}

Time-step refinement from 0.004 to 0.001 reduces the Burgers adaptive $L^2$ error from $3.64\times10^{-3}$ to $1.21\times10^{-3}$. The Allen–Cahn error decreases from $2.08\times10^{-3}$ to $1.15\times10^{-3}$ and then begins to approach a spatial or adaptation floor. The fixed circles exhibit much weaker time-step sensitivity because spatial representation error dominates. These observations do not establish a uniform convergence order of the combined adaptive method.

## 8.4 Heterogeneous transient diffusion

**Table 5. Heterogeneous transient diffusion at final time 0.1 and a time step of 0.005. The FDM error uses the interior grid norm, while neural errors use independent quadrature.**

| Method | N | Relative $L^2$ | Relative energy | Time s |
| :--- | :--- | :--- | :--- | :--- |
| Adaptive circular | 129 | 9.14e-04 | 1.49e-02 | 6.227 |
| Fixed circular | 129 | 1.12e-02 | 8.63e-02 | 0.994 |
| Conservative FDM | 6241 | 3.35e-04 | Not evaluated | 0.035 |



At Δt = 0.005, the adaptive transient $L^2$ error is $9.14\times10^{-4}$, compared with $1.12\times10^{-2}$ for the fixed network. The conservative FDM error is $3.35\times10^{-4}$. Refining the time step changes the circular error only slightly, which is consistent with a dominant spatial contribution. The geometry is selected only at the first step, and this decaying spatial profile does not test repeated geometry adaptation over a long time horizon.

![Figure 6. Time-step refinement for all three evolutionary cases. The adaptive and fixed spaces have the same coefficient budgets within each case. Flattening of a curve indicates a remaining spatial or adaptation contribution.](figures/figure_6_time_refinement.png){width=6.3in}

## 8.5 Energy controlled compression

**Table 6. Energy-guarded pruning of the 129-coefficient adaptive solution. τ is relative to solution energy, not to the unknown approximation error.**

| Case | τ | Retained N | Relative $L^2$ | Relative energy |
| :--- | :--- | :--- | :--- | :--- |
| Poisson | 1.00e-08 | 109 | 7.04e-05 | 7.34e-04 |
| Poisson | 1.00e-06 | 39 | 1.66e-04 | 1.22e-03 |
| Poisson | 1.00e-04 | 8 | 1.85e-03 | 6.06e-03 |
| Interface | 1.00e-08 | 127 | 1.00e-02 | 5.72e-02 |
| Interface | 1.00e-06 | 121 | 9.98e-03 | 5.72e-02 |
| Interface | 1.00e-04 | 88 | 1.07e-02 | 5.79e-02 |



The smooth solution tolerates more compression than the localized interface solution. With τ = $10^{-8}$, pruning reduces the smooth network from 129 to 109 coefficients while changing relative $L^2$ error from $6.92\times10^{-5}$ to $7.04\times10^{-5}$. The same budget removes only two coefficients from the interface network. At τ = $10^{-4}$, the smooth network collapses to its eight global functions and its error rises to $1.85\times10^{-3}$. This behavior is consistent with the interpretation of the budget: it bounds an energy increase relative to solution energy, not the original approximation error. A small increase relative to the solution can be large relative to an already very small error.

## 8.6 Quadrature and flux majorant checks

**Table 7. Selected-space integration check. The selected 129-dimensional space is held fixed and refitted with different training quadrature resolutions. All errors use the same independent evaluation rule.**

| Case | Subdivisions | Relative $L^2$ | Relative energy |
| :--- | :--- | :--- | :--- |
| Poisson | 32 | 7.29e-05 | 7.38e-04 |
| Poisson | 64 | 6.92e-05 | 7.27e-04 |
| Poisson | 96 | 6.93e-05 | 7.27e-04 |
| Interface | 32 | 1.33e-02 | 7.66e-02 |
| Interface | 64 | 1.00e-02 | 5.72e-02 |
| Interface | 96 | 1.00e-02 | 5.71e-02 |



For a fixed selected space, increasing the integration resolution from 64 to 96 subdivisions produces very similar field errors, although interface coefficients are more sensitive than the final error norms. The scaled condition numbers of the selected stiffness matrices are approximately 5.07 × $10^{3}$ for Poisson and 4.04 × $10^{3}$ for the interface case. Independent norms are a more informative check than coefficient differences alone. The underresolved 32-subdivision interface fit has a noticeably larger error, confirming that structural adaptation must be accompanied by adequate integration.

**Table 8. Quadrature-evaluated functional majorants from a separate 24 × 24 RT0 flux grid. Effectivity is the estimated majorant divided by the measured absolute energy error.**

| Case | Energy error | Majorant | Effectivity | Flux unknowns |
| :--- | :--- | :--- | :--- | :--- |
| Poisson | 1.61e-03 | 2.03e-01 | 125.5 | 1200 |
| Interface | 2.53e-02 | 3.24e+00 | 128.0 | 1200 |



The RT0 reconstruction on a 24 × 24 flux grid produces majorant effectivities of approximately 125 and 128. The equilibrium residual dominates, especially where the manufactured forcing changes rapidly within a flux cell. The result is compatible with a continuous upper-bound framework but is far too pessimistic to be an effective tolerance controller. A higher-order or locally refined flux space, with certified integration if a numerical guarantee is required, is necessary before making an operational certification claim.

## 8.7 Radius-sensitivity experiment

The active 129-coefficient circular hierarchy was reselected after multiplying all Wendland support radii by the indicated common factor. Centers, hierarchy, 64-subdivision training quadrature, and independent 80-subdivision error quadrature were held fixed. This controlled experiment measures support-width sensitivity and does not replace the original benchmark or establish robust automatic radius optimization.

**Table 9. Radius sensitivity at 129 coefficients; deterministic single runs.**

| Problem | Radius factor | Relative L2 | Relative energy | Scaled condition |
|:---|---:|---:|---:|---:|
| Poisson | 1.0 | 6.919e-05 | 7.270e-04 | 5.067e+03 |
| Poisson | 1.2 | 2.226e-05 | 2.731e-04 | 5.453e+04 |
| Interface | 1.0 | 1.000e-02 | 5.719e-02 | 4.040e+03 |
| Interface | 1.2 | 5.737e-03 | 3.294e-02 | 1.788e+04 |
| Interface | 1.5 | 3.016e-03 | 2.077e-02 | 3.739e+05 |

The relative L2 error for smooth Poisson diffusion decreases from 6.92e-5 at scale 1.0 to 2.23e-5 at scale 1.2; the corresponding scaled condition number increases from 5.07e3 to 5.45e4. For interface diffusion, scale 1.5 decreases the relative L2 error from 1.00e-2 to 3.02e-3 but raises the scaled condition number from 4.04e3 to 3.74e5. Scale 1.2 is an intermediate interface trade-off, with relative L2 error 5.74e-3 and scaled condition number 1.79e4. These results highlight a tunable accuracy-versus-conditioning trade-off, not a general improvement in stability or wall-clock speed.

# 9 Verification reproducibility and limitations

## 9.1 Software verification

Seven automated checks accompany the study. They verify first and second basis derivatives against centered differences, exact zero boundary traces, the analytic parameter Jacobian of the tanh residual, continuity of the manufactured interface trace and flux, the elliptic refinement identity and selected-system residual, manufactured sources for three diffusion fields, and the Burgers travelling-wave equation. All seven checks pass in the supplied execution environment. They target independent mathematical consequences rather than merely checking that a routine reproduces its own output.

Numerical error norms use a different integration rule from the training rule. P1 FEM is assembled independently on triangles, while flux recovery uses a separate rectangular $H(\operatorname{div})$ space. The one-dimensional and transient finite difference solvers do not reuse circular basis operators. These separations reduce shared implementation errors, although they do not constitute a formal verification of the software.

The supplementary package includes complete source files, requirements, numerical JSON records, saved coefficient arrays, plotting code and the editable manuscript source. The environment record identifies the Python, NumPy and BLAS configuration. Every table and figure is assembled from those saved results. The study requires no GPU or neural framework, and geometry selection is deterministic under the stated hierarchy and integration rules.

## 9.2 Limits of the mathematical claims

The elliptic projection, block-gain, removal and flux results are proved under explicit assumptions. The unrestricted density argument does not establish a rate for the implemented radius-to-spacing relation. The analysis does not show estimator efficiency, robust constants for arbitrarily large coefficient contrasts, or contraction of Algorithm 1. The nonlinear residual score is a heuristic extension and has no transferred exact-gain theorem. These distinctions prevent the experimental success of a few cases from being used as a substitute for an adaptive convergence proof.

For general interfaces, a fitted coordinate such as Eq. (24) may be unavailable. A piecewise trial space, weak interface coupling or mixed flux formulation would then require additional analysis and implementation. Similarly, the boundary multiplier used here is simple on boxes; complex domains introduce geometric approximation and trace-enforcement issues that are not represented in the experiments.

## 9.3 Limits of the comparisons

The neural comparison uses one hidden layer, one optimizer configuration and three seeds. It is deliberately transparent but does not exhaust the performance of deep PINNs, radial networks or neural domain decomposition. In particular, the most closely related AB-PINN and sparse RBF methods [17,18] are discussed but not directly implemented. The fixed strong-circle baseline also changes the formulation relative to the adaptive Ritz method, so the fixed Ritz baseline is the relevant control for isolating structural adaptation.

Classical baselines use regular grids. Adaptive FEM could concentrate degrees of freedom near the same local structures and may change the error-versus-work comparison substantially. No high-dimensional PDE, inverse problem, irregular geometry, long-time chaotic evolution or noisy observation test is included. Peak memory and hardware energy are not benchmarked. The current dense candidate matrix and all-feature evaluation limit practical scaling even though the radial functions have compact support.

# 10 Conclusions

We have specified an adaptive circular physics-informed discretization based on compact radial features and a weak Ritz formulation. For symmetric diffusion, residual Schur-complement scores give the exact energy reduction associated with an admissible hierarchical split under exact coefficient solution and integration. A corresponding constrained quadratic calculation gives an energy-based pruning rule. The analysis also separates approximation, quadrature and algebraic effects, and places a classical functional majorant alongside the structural selection criterion.

The radius-sensitivity check additionally shows that larger circular supports improve the two tested elliptic errors at unchanged coefficient count, but may substantially increase the selected-system condition number. The numerical evidence supports a selective conclusion. At matched output coefficient counts, adaptive circles resolve the localized interface and moving-front examples more accurately than the tested fixed circular spaces. A smooth global problem is better served by the tested tanh PINNs, and the independent classical methods remain faster. The coarse recovered-flux majorant is reliable only as a continuous framework and is numerically too pessimistic for efficient stopping in these tests. These outcomes identify both the value and the current limits of the method.

The most useful next steps are a stable approximation theory for the actual dyadic hierarchy, a sharper flux reconstruction, local sparse candidate assembly, and direct equal-work comparisons with adaptive FEM and recent adaptive radial neural solvers. Those developments would determine whether the present representation advantage can become a broader numerical-method advantage.

# Data and code availability

The full implementation, numerical records, saved coefficients and figure-generation scripts are supplied with the manuscript as supplementary material. All fields are analytic or manufactured; no restricted third-party dataset is needed. The package README documents the commands that regenerate the reported study.

# Appendix A Derivatives and coefficient updates

For $s=\|x-z\|/r$ within the unit support radius, the radial derivatives are

$$
\psi'(s)=-20s(1-s)^3,\qquad
\psi''(s)=-20(1-s)^2(1-4s).\qquad (A1)
$$

For $\phi(x)=\psi(\|x-z\|/r)$, differentiation gives

$$
\nabla\phi=-\frac{20}{r^2}(1-s)^3(x-z),\qquad
\Delta\phi=-\frac{20}{r^2}(1-s)^2\{d-(d+3)s\}.\qquad (A2)
$$

Both formulas are zero outside the support, with their continuous center values understood by limits. For the boundary-conforming feature Bφ, the gradient is $\phi\nabla B+B\nabla\phi$ and the Laplacian is $\phi\Delta B+2\nabla B\cdot\nabla\phi+B\Delta\phi$. The piecewise interface map is applied through the chain rule inside each material region. Its derivative jump is treated through flux continuity, not by differentiating a discontinuous coefficient as an ordinary smooth function.

For incremental selection, let L store energy-orthogonalization factors for all candidate features against the selected features. A new pivot p has residual column $K_{\cdot p}-LL_p^T$ and squared norm $\sigma_p$. Division by $\sqrt{\sigma_p}$ yields a new factor column. Updating the candidate load residual by its projection onto that column permits subsequent child-block scores to be evaluated without resolving every augmented coefficient problem. The supplied implementation checks the resulting energy difference against Eq. (15).

# Appendix B Conservative circular approximation estimate

Normalize $k(x)=C\psi(\|x\|)$ so that its integral is one and define $k_\rho(x)=\rho^{-d}k(x/\rho)$. Radial symmetry makes its first moments vanish. For a smooth compactly supported q, Taylor’s formula in $H^1$ gives

$$
\|q-k_\rho*q\|_{H^1}\leq C\rho^2\|q\|_{H^3}.\qquad (B1)
$$

Partition a fixed interior box containing the support of q into cells of maximum diameter h, with centers $c_j$, and approximate the convolution integral by $s_{\rho,h}(x)=\sum_j|K_j|q(c_j)k_\rho(x-c_j)$. The difference between an integrand and its cell-center value has two terms: variation of q, and translation of $k_\rho$. Minkowski’s inequality and the translation estimate in $H^1$ bound their sum by

$$
Ch\left(\|\nabla q\|_\infty\|k_\rho\|_{H^1}
+\|q\|_\infty\|\nabla k_\rho\|_{H^1}\right).\qquad (B2)
$$

Scaling gives $\|k_\rho\|_{H^1}\leq C\rho^{-d/2-1}$ and $\|\nabla k_\rho\|_{H^1}\leq C\rho^{-d/2-2}$ for 0 < ρ ≤ 1. The profile is $C^2$ and compactly supported, so the required $H^2$ kernel norm is finite. Multiplication by the smooth boundary factor is bounded in $H^1$. Combining these observations with Eq. (B1) yields Eq. (16). The estimate uses a fixed smooth quotient supported away from the boundary and is not asserted uniformly for all functions in $H_0^1$ or for a fixed-width geometric hierarchy.

# References

[1] I. E. Lagaris, A. Likas, D. I. Fotiadis, Artificial neural networks for solving ordinary and partial differential equations, IEEE Transactions on Neural Networks 9 (1998) 987–1000. [Link](https://doi.org/10.1109/72.712178)

[2] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (2019) 686–707. [Link](https://doi.org/10.1016/j.jcp.2018.10.045)

[3] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang, Physics-informed machine learning, Nature Reviews Physics 3 (2021) 422–440. [Link](https://doi.org/10.1038/s42254-021-00314-5)

[4] S. Wang, Y. Teng, P. Perdikaris, Understanding and mitigating gradient flow pathologies in physics-informed neural networks, SIAM Journal on Scientific Computing 43 (2021) A3055–A3081. [Link](https://doi.org/10.1137/20M1318043)

[5] S. Wang, X. Yu, P. Perdikaris, When and why PINNs fail to train: A neural tangent kernel perspective, Journal of Computational Physics 449 (2022) 110768. [Link](https://doi.org/10.1016/j.jcp.2021.110768)

[6] J. Yu, L. Lu, X. Meng, G. E. Karniadakis, Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems, Computer Methods in Applied Mechanics and Engineering 393 (2022) 114823. [Link](https://doi.org/10.1016/j.cma.2022.114823)

[7] C. Wu, M. Zhu, Q. Tan, Y. Kartha, L. Lu, A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks, Computer Methods in Applied Mechanics and Engineering 403 (2023) 115671. [Link](https://doi.org/10.1016/j.cma.2022.115671)

[8] B. Gao, R. Yao, Y. Li, Physics-informed neural networks with adaptive loss weighting algorithm for solving partial differential equations, Computers & Mathematics with Applications 181 (2025). [Link](https://doi.org/10.1016/j.camwa.2025.01.007)

[9] X. Yang, M. Ran, H. Pu, A novel adaptive gradient-enhanced physics-informed neural networks for solving partial differential equations, Computers & Mathematics with Applications 213 (2026) 242–256. [Link](https://doi.org/10.1016/j.camwa.2026.04.026)

[10] W. E, B. Yu, The Deep Ritz Method: A deep learning-based numerical algorithm for solving variational problems, Communications in Mathematics and Statistics 6 (2018) 1–12. [Link](https://doi.org/10.1007/s40304-018-0127-z)

[11] E. Kharazmi, Z. Zhang, G. E. M. Karniadakis, hp-VPINNs: Variational physics-informed neural networks with domain decomposition, Computer Methods in Applied Mechanics and Engineering 374 (2021) 113547. [Link](https://doi.org/10.1016/j.cma.2020.113547)

[12] S. Berrone, C. Canuto, M. Pintore, Variational physics informed neural networks: The role of quadratures and test functions, Journal of Scientific Computing 92 (2022) 100. [Link](https://doi.org/10.1007/s10915-022-01950-4)

[13] S. Berrone, C. Canuto, M. Pintore, Solving PDEs by variational physics-informed neural networks: An a posteriori error analysis, Annali dell’Università di Ferrara 68 (2022) 575–595. [Link](https://doi.org/10.1007/s11565-022-00441-6)

[14] S. Rojas, P. Maczuga, J. Muñoz-Matute, D. Pardo, M. Paszyński, Robust Variational Physics-Informed Neural Networks, Computer Methods in Applied Mechanics and Engineering 425 (2024) 116904. [Link](https://doi.org/10.1016/j.cma.2024.116904)

[15] B. Moseley, A. Markham, T. Nissen-Meyer, Finite basis physics-informed neural networks (FBPINNs): A scalable domain decomposition approach for solving differential equations, Advances in Computational Mathematics 49 (2023) 62. [Link](https://doi.org/10.1007/s10444-023-10065-9)

[16] J. Bai, G.-R. Liu, A. Gupta, L. Alzubaidi, X.-Q. Feng, Y. Gu, Physics-informed radial basis network (PIRBN): A local approximating neural network for solving nonlinear partial differential equations, Computer Methods in Applied Mechanics and Engineering 415 (2023) 116290. [Link](https://doi.org/10.1016/j.cma.2023.116290)

[17] J. Botvinick-Greenhouse, W. H. Ali, M. Benosman, S. Mowlavi, AB-PINNs: Adaptive-basis physics-informed neural networks for residual-driven domain decomposition, Machine Learning: Science and Technology 7 (2026) 045025. [Link](https://doi.org/10.1088/2632-2153/ae8638)

[18] Z. Shao, K. Pieper, X. Tian, Solving nonlinear PDEs with sparse radial basis function networks, Journal of Machine Learning Research 27 (2026) 177, 1–58. [Link](https://www.jmlr.org/beta/papers/v27/25-1399.html)

[19] J. W. Siegel, Q. Hong, X. Jin, W. Hao, J. Xu, Greedy training algorithms for neural networks and applications to PDEs, Journal of Computational Physics 484 (2023) 112084. [Link](https://doi.org/10.1016/j.jcp.2023.112084)

[20] H. Wendland, Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree, Advances in Computational Mathematics 4 (1995) 389–396. [Link](https://doi.org/10.1007/BF02123482)

[21] H. Wendland, Scattered Data Approximation, Cambridge Monographs on Applied and Computational Mathematics 17, Cambridge University Press, Cambridge, 2005. [Link](https://www.cambridge.org/core/books/scattered-data-approximation/980EEC9DBC4CAA711D089187818135E3)

[22] R. Schaback, Error estimates and condition numbers for radial basis function interpolation, Advances in Computational Mathematics 3 (1995) 251–264. [Link](https://doi.org/10.1007/BF02432002)

[23] I. Babuška, J. M. Melenk, The partition of unity method, International Journal for Numerical Methods in Engineering 40 (1997) 727–758. [Link](https://doi.org/10.1002/(SICI)1097-0207(19970228)40:4<727::AID-NME86>3.0.CO;2-N)

[24] Y. Shin, J. Darbon, G. E. Karniadakis, On the convergence of physics informed neural networks for linear second-order elliptic and parabolic type PDEs, Communications in Computational Physics 28 (2020) 2042–2074. [Link](https://doi.org/10.4208/cicp.OA-2020-0193)

[25] M. Zeinhofer, R. Masri, K.-A. Mardal, A unified framework for the error analysis of physics-informed neural networks, IMA Journal of Numerical Analysis 45 (2025) 2988–3025. [Link](https://doi.org/10.1093/imanum/drae081)

[26] T. De Ryck, S. Mishra, Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning, Acta Numerica 33 (2024) 633–713. [Link](https://doi.org/10.1017/S0962492923000089)

[27] L. Ernst, K. Urban, A certified wavelet-based physics-informed neural network for the solution of parameterized partial differential equations, IMA Journal of Numerical Analysis 45 (2025) 494–515. [Link](https://doi.org/10.1093/imanum/drae011)

[28] S. I. Repin, A posteriori error estimation for variational problems with uniformly convex functionals, Mathematics of Computation 69 (2000) 481–500. [Link](https://doi.org/10.1090/S0025-5718-99-01190-4)

[29] W. Dörfler, A convergent adaptive algorithm for Poisson’s equation, SIAM Journal on Numerical Analysis 33 (1996) 1106–1124. [Link](https://doi.org/10.1137/0733054)

[30] N. Sukumar, A. Srivastava, Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks, Computer Methods in Applied Mechanics and Engineering 389 (2022) 114333. [Link](https://doi.org/10.1016/j.cma.2021.114333)

[31] P. A. Raviart, J. M. Thomas, A mixed finite element method for 2-nd order elliptic problems, in I. Galligani, E. Magenes (Eds.), Mathematical Aspects of Finite Element Methods, Lecture Notes in Mathematics 606, Springer, Berlin, 1977, pp. 292–315. [Link](https://doi.org/10.1007/BFb0064470)

[32] C. J. Roy, C. C. Nelson, T. M. Smith, C. C. Ober, Verification of Euler/Navier–Stokes codes using the method of manufactured solutions, International Journal for Numerical Methods in Fluids 44 (2004) 599–620. [Link](https://doi.org/10.1002/fld.660)

[33] S. M. Allen, J. W. Cahn, A microscopic theory for antiphase boundary motion and its application to antiphase domain coarsening, Acta Metallurgica 27 (1979) 1085–1095. [Link](https://doi.org/10.1016/0001-6160(79)90196-2)

[34] J. D. Cole, On a quasi-linear parabolic equation occurring in aerodynamics, Quarterly of Applied Mathematics 9 (1951) 225–236. [Link](https://doi.org/10.1090/qam/42889)
