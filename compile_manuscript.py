"""Fill measured tables and construct editable native Word mathematical content."""
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output';OUT.mkdir(exist_ok=True)
R=ROOT/'results'

def load(name):return json.loads((R/name).read_text())
def num(x):return f'{x:.2e}'
def table(headers,rows):
    return '\n'+'| '+' | '.join(headers)+' |\n| '+ ' | '.join([':---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'

def main():
    text=(ROOT/'manuscript_template.md').read_text()
    text=text.replace(r'{\rm new}',r'{\mathrm{new}}').replace(r'{\rm alg}',r'{\mathrm{alg}}')
    chunks=[]
    for no,name,label in [(2,'poisson','Smooth Poisson'),(3,'interface','Interface diffusion')]:
        data=load(name+'.json');rows=[]
        for q in [next(r for r in data['adaptive'] if r['size']==129),data['fixed'],data['strong']]:
            rows.append([q['method'],q['size'],num(q['relative_l2']),num(q['relative_energy']),f"{q['seconds']:.3f}"])
        pins=load(name+'_pinn.json')
        for method in ['tanh PINN','RAR tanh PINN','FD gPINN']:
            rs=[q for q in pins if q['method']==method]
            stat=lambda k:f"{np.mean([q[k] for q in rs]):.2e} ± {np.std([q[k] for q in rs],ddof=1):.2e}"
            rows.append([method,129,stat('relative_l2'),stat('relative_energy'),f"{np.mean([q['seconds'] for q in rs]):.3f}"])
        for q in data['fem']:rows.append([q['method'],q['size'],num(q['relative_l2']),num(q['relative_energy']),f"{q['seconds']:.3f}"])
        chunks.append(f'**Table {no}. {label} results. Neural optimization rows give mean ± sample standard deviation over three seeds; time is the mean. Other rows are deterministic single runs. N is the fitted coefficient or classical interior-unknown count.**\n'+table(['Method','N','Relative L2','Relative energy','Time s'],rows))
    text=text.replace('{{ELLIPTIC_TABLES}}','\n'.join(chunks))
    evo=load('evolution.json');rows=[]
    for case,label in [('allen_cahn','Allen–Cahn'),('burgers','Burgers')]:
        for method in ['Adaptive circular','Fixed circular','Central FDM']:
            q=next(q for q in evo if q['case']==case and q['method']==method and q['dt']==.002)
            rows.append([label,method,q['size'],num(q['relative_l2']),num(q['relative_gradient']),f"{q['seconds']:.3f}"])
    text=text.replace('{{FRONT_TABLE}}','**Table 4. Nonlinear front results at a time step of 0.002. Gradient errors are relative H1 seminorm errors.**\n'+table(['Case','Method','N','Relative L2','Relative gradient','Time s'],rows))
    rows=[]
    for q in evo:
        if q['case']=='transient' and q['dt']==.005:
            rows.append([q['method'],q['size'],num(q['relative_l2']),num(q['relative_energy']) if 'relative_energy' in q else 'Not evaluated',f"{q['seconds']:.3f}"])
    text=text.replace('{{TRANSIENT_TABLE}}','**Table 5. Heterogeneous transient diffusion at final time 0.1 and a time step of 0.005. The FDM error uses the interior grid norm, while neural errors use independent quadrature.**\n'+table(['Method','N','Relative L2','Relative energy','Time s'],rows))
    rows=[]
    for name,label in [('poisson','Poisson'),('interface','Interface')]:
        for q in load(name+'.json')['pruning']:
            rows.append([label,num(q['tolerance_squared_relative']),q['size'],num(q['relative_l2']),num(q['relative_energy'])])
    text=text.replace('{{PRUNING_TABLE}}','**Table 6. Energy-guarded pruning of the 129-coefficient adaptive solution. τ is relative to solution energy, not to the unknown approximation error.**\n'+table(['Case','τ','Retained N','Relative L2','Relative energy'],rows))
    rows=[]
    for name,label in [('poisson','Poisson'),('interface','Interface')]:
        for q in load(name+'.json')['quadrature_check']['levels']:
            rows.append([label,q['subdivisions'],num(q['relative_l2']),num(q['relative_energy'])])
    text=text.replace('{{QUADRATURE_TABLE}}','**Table 7. Selected-space integration check. The selected 129-dimensional space is held fixed and refitted with different training quadrature resolutions. All errors use the same independent evaluation rule.**\n'+table(['Case','Subdivisions','Relative L2','Relative energy'],rows))
    rows=[]
    for name,label in [('poisson','Poisson'),('interface','Interface')]:
        data=load(name+'.json');q=data['majorant'];err=next(r['energy_error'] for r in data['adaptive'] if r['size']==129)
        rows.append([label,num(err),num(q['majorant']),f"{q['effectivity']:.1f}",q['flux_dofs']])
    text=text.replace('{{MAJORANT_TABLE}}','**Table 8. Quadrature-evaluated functional majorants from a separate 24 × 24 RT0 flux grid. Effectivity is the estimated majorant divided by the measured absolute energy error.**\n'+table(['Case','Energy error','Majorant','Effectivity','Flux unknowns'],rows))
    references=json.loads((ROOT/'references.json').read_text())
    citation_pattern=r'\[(\d+(?:\s*[,–-]\s*\d+)*)\]'
    citation_order=[]
    def expand(group):
        values=[]
        for part in group.split(','):
            bounds=re.split('[–-]',part.strip())
            values.extend(range(int(bounds[0]),int(bounds[-1])+1))
        return values
    for hit in re.finditer(citation_pattern,text):
        for n in expand(hit[1]):
            if n not in citation_order:citation_order.append(n)
    for q in references:
        if q['n'] not in citation_order:citation_order.append(q['n'])
    renumber={old:new for new,old in enumerate(citation_order,1)}
    text=re.sub(citation_pattern,lambda m:'['+','.join(str(renumber[n]) for n in expand(m[1]))+']',text)
    refmap={q['n']:q for q in references}
    text=text.replace('{{REFERENCES}}','\n\n'.join(f"[{renumber[n]}] {refmap[n]['text']} [Link]({refmap[n]['url']})" for n in citation_order))
    # Convert compact mathematical prose to native editable inline Word math.
    replace={
      'ut = ε2uxx+u−u3+s':r'$u_t=\epsilon^2u_{xx}+u-u^3+s$',
      'ut+uux = νuxx':r'$u_t+uu_x=\nu u_{xx}$',
      'f = −u−div(a∇u)':r'$f=-u-\nabla\cdot(a\nabla u)$',
      'a(x,y) = 1+0.5 sin(2πx)sin(2πy)':r'$a(x,y)=1+0.5\sin(2\pi x)\sin(2\pi y)$',
      'B(x,y) = 16x(1−x)y(1−y)':r'$B(x,y)=16x(1-x)y(1-y)$',
      'B(x) = 4x(1−x)':r'$B(x)=4x(1-x)$',
      'hρ−d/2−2':r'$h\rho^{-d/2-2}$',
      'rCTHC−1rC':r'$r_C^TH_C^{-1}r_C$',
      'HC−1rC':r'$H_C^{-1}r_C$',
      'A(uS−vQ,z)':r'$A(u_S-v_Q,z)$',
      'uS−vQ':r'$u_S-v_Q$',
      'u−uS':r'$u-u_S$',
      'u−vS':r'$u-v_S$',
      'u = uε':r'$u=u_\varepsilon$',
      'H(div, Ω)':r'$H(\operatorname{div},\Omega)$',
      'H(div)':r'$H(\operatorname{div})$',
      'O(D2)':r'$O(D^2)$','O(DM2)':r'$O(DM^2)$','O(M3)':r'$O(M^3)$',
      'O(DM)':r'$O(DM)$','O(NM)':r'$O(NM)$',
      'L-infinity':r'$L^\infty$',
    }
    for a,b in replace.items():text=text.replace(a,b)
    inline_formulas={
        'δc = −cjKS−1ej/(KS−1)jj':r'\delta c=-c_jK_S^{-1}e_j/(K_S^{-1})_{jj}',
        'δcTKSδc':r'\delta c^T K_S\delta c',
        'δcj = −cj':r'\delta c_j=-c_j',
        'Rv(z) = F(z)−A(v,z)':r'R_v(z)=F(z)-A(v,z)',
        'A(u,e) = F(e)':r'A(u,e)=F(e)',
        'A(u,z) = F(z)':r'A(u,z)=F(z)',
        'A(uS−vQ,z) = (F−FQ)(z)+(AQ−A)(vQ,z)+FQ(z)−AQ(vQ,z)':r'A(u_S-v_Q,z)=(F-F_Q)(z)+(A_Q-A)(v_Q,z)+F_Q(z)-A_Q(v_Q,z)',
        '(F−FQ)(z)+(AQ−A)(vQ,z)+FQ(z)−AQ(vQ,z)':r'(F-F_Q)(z)+(A_Q-A)(v_Q,z)+F_Q(z)-A_Q(v_Q,z)',
        '2/(3·2^l)':r'2/(3\,2^\ell)',
        '3·2^l':r'3\,2^\ell',
        'q = uε/B':r'q=u_\varepsilon/B',
        'kρ ∗ q → q':r'k_\rho*q\to q',
        'CP/√amin':r'C_P/\sqrt{a_{\min}}',
        'CP = 1/(π√2)':r'C_P=1/(\pi\sqrt{2})',
        '(1/Δt−1)':r'(1/\Delta t-1)',
        'S0 = 1/2+1/(2κ)':r'S_0=1/2+1/(2\kappa)',
        'C0 = (1/8+3/(8κ))/S0':r'C_0=(1/8+3/(8\kappa))/S_0',
        'U(x) = ∫0x(C0−s)/a(s) ds':r'U(x)=\int_0^x(C_0-s)/a(s)\,ds',
        'a∂xu = S0−1∂ξu':r'a\partial_xu=S_0^{-1}\partial_\xi u',
        '−(aU′)′ = 1':r'-(aU\prime)\prime=1',
        's = ‖x−z‖/r':r's=\|x-z\|/r',
        'φ(x) = ψ(‖x−z‖/r)':r'\phi(x)=\psi(\|x-z\|/r)',
        'φΔB+2∇B·∇φ+BΔφ':r'\phi\Delta B+2\nabla B\cdot\nabla\phi+B\Delta\phi',
        'φ∇B+B∇φ':r'\phi\nabla B+B\nabla\phi',
        'e = v−u':r'e=v-u',
        'e = u−v':r'e=u-v',
        'K·p−LLpT':r'K_{\cdot p}-LL_p^T',
        '√σp':r'\sqrt{\sigma_p}',
        'k(x) = Cψ(‖x‖)':r'k(x)=C\psi(\|x\|)',
        'kρ(x) = ρ−d k(x/ρ)':r'k_\rho(x)=\rho^{-d}k(x/\rho)',
        'sρ,h(x) = Σj |Kj|q(cj)kρ(x−cj)':r's_{\rho,h}(x)=\sum_j|K_j|q(c_j)k_\rho(x-c_j)',
        '‖kρ‖H1 ≤ Cρ−d/2−1':r'\|k_\rho\|_{H^1}\leq C\rho^{-d/2-1}',
        '‖∇kρ‖H1 ≤ Cρ−d/2−2':r'\|\nabla k_\rho\|_{H^1}\leq C\rho^{-d/2-2}',
    }
    inline_tokens={'vnew':r'v_{\mathrm{new}}','vS':r'v_S','VS':r'V_S','uS':r'u_S','vQ':r'v_Q',
        'AQ':r'A_Q','FQ':r'F_Q','HC':r'H_C','KS':r'K_S','rC':r'r_C','δQ':r'\delta_Q',
        'δalg':r'\delta_{\mathrm{alg}}','uε':r'u_\varepsilon','kρ':r'k_\rho','sρ,h':r's_{\rho,h}',
        'δc':r'\delta c','φi':r'\phi_i','σp':r'\sigma_p','cj':r'c_j','CP':r'C_P','ε2':r'\epsilon^2'}
    def prose_math(segment):
        for a,b in inline_formulas.items():segment=segment.replace(a,'$'+b+'$')
        # Protect any newly inserted equations before converting remaining tokens.
        chunks=re.split(r'(\$[^$]*\$)',segment)
        for i in range(0,len(chunks),2):
            for a,b in inline_tokens.items():
                chunks[i]=re.sub(r'(?<![\w])'+re.escape(a)+r'(?![\w])',lambda m:'$'+b+'$',chunks[i])
        return ''.join(chunks)
    # Exponents in prose are converted without changing displayed equations or URLs.
    lines=[];inmath=False
    for line in text.splitlines():
        if line.strip()=='$$':inmath=not inmath;lines.append(line);continue
        if not inmath and not line.startswith('['):
            parts=re.split(r'(\$[^$]*\$)',line)
            for i in range(0,len(parts),2):parts[i]=prose_math(parts[i])
            line=''.join(parts)
            line=re.sub(r'(\d+(?:\.\d+)?) × 10−(\d+)',lambda m:'$'+m[1]+r'\times10^{-'+m[2]+'}$',line)
            line=re.sub(r'(?<![\w\\])10−(\d+)',lambda m:'$10^{-'+m[1]+'}$',line)
            line=re.sub(r'(?<![\w\\])10\^(\d+)',lambda m:'$10^{'+m[1]+'}$',line)
            for a,b in [('H01',r'H_0^1'),('H1',r'H^1'),('H2',r'H^2'),('H3',r'H^3'),('L2',r'L^2'),('C2',r'C^2')]:
                line=re.sub(r'\b'+a+r'\b',lambda m:'$'+b+'$',line)
        lines.append(line)
    text='\n'.join(lines)+'\n'
    if '{{' in text:raise RuntimeError('Unfilled manuscript data')
    source=ROOT/'manuscript.md';source.write_text(text)
    raw=OUT/'AC_PINN_native_raw.docx';final=OUT/'AC_PINN_Manuscript.docx'
    subprocess.run(['pandoc',str(source),'--from=markdown+tex_math_dollars','--to=docx','--resource-path='+str(ROOT),'--output',str(raw)],check=True,cwd=ROOT)
    d=Document(raw)
    sec=d.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=Inches(.75);sec.bottom_margin=Inches(.75);sec.left_margin=Inches(.8);sec.right_margin=Inches(.8)
    for name in ['Normal','Body Text','First Paragraph']:
        s=d.styles[name];s.font.name='Times New Roman';s.font.size=Pt(11);s.font.color.rgb=RGBColor(0,0,0)
        s.paragraph_format.space_after=Pt(6);s.paragraph_format.line_spacing=1.12
        s.paragraph_format.widow_control=True
    for name,size in [('Title',19),('Subtitle',12),('Heading 1',14),('Heading 2',12),('Heading 3',11)]:
        s=next((s for s in d.styles if s.name.lower()==name.lower()),None)
        if s is None:s=d.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
        s.font.name='Times New Roman';s.font.size=Pt(size);s.font.color.rgb=RGBColor(0,0,0)
        s.paragraph_format.keep_with_next=True;s.paragraph_format.space_before=Pt(12);s.paragraph_format.space_after=Pt(6)
        if name in ['Title','Subtitle']:s.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
        if name.startswith('Heading'):s.font.bold=True
        ppr=s.element.find(qn('w:pPr'))
        if ppr is not None:
            border=ppr.find(qn('w:pBdr'))
            if border is not None:ppr.remove(border)
    caption_style=next(s for s in d.styles if s.name.lower()=='caption')
    caption_style.font.name='Times New Roman';caption_style.font.size=Pt(10)
    caption_style.font.color.rgb=RGBColor(0,0,0)
    for p in d.paragraphs:
        if p.text.startswith('Figure '):
            p.style=caption_style;p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(9)
            p.paragraph_format.keep_with_next=False
        if p.text.startswith('Table '):
            p.paragraph_format.keep_with_next=True;p.paragraph_format.space_before=Pt(7)
        if p.style.name.startswith('Heading'):
            for run in p.runs:run.font.color.rgb=RGBColor(0,0,0);run.font.name='Times New Roman'
        if p.style.name in ['Title','Subtitle']:
            for run in p.runs:run.font.name='Times New Roman'
        if p._p.xpath('.//w:drawing'):
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
        if p._p.xpath('.//m:oMathPara'):
            p.paragraph_format.space_before=Pt(5);p.paragraph_format.space_after=Pt(7)
            p.paragraph_format.keep_together=True
    # Keep the nine-step algorithm together.
    in_algorithm=False
    for p in d.paragraphs:
        if p.text.startswith('Algorithm 1.'):
            in_algorithm=True;p.paragraph_format.keep_with_next=True
        elif in_algorithm and p.style.name=='Compact':
            p.paragraph_format.keep_with_next=True
            if p.text.startswith('Reevaluate'):p.paragraph_format.keep_with_next=False;in_algorithm=False
        elif in_algorithm and p.text.startswith('An error tolerance'):in_algorithm=False
    for t in d.tables:
        t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        ncols=len(t.columns)
        widths=([1.8,.5,1.8,1.8,.8] if ncols==5 else [1.0,1.7,.45,1.2,1.4,.75] if ncols==6 else [1.2,1.2,2.15,2.15] if ncols==4 else [1.55,2.25,3.0])
        if t.cell(0,0).text=='Case' and ncols==5:widths=[1.0,1.2,1.6,1.4,1.5]
        if t.cell(0,0).text=='Case' and t.cell(0,1).text=='τ':widths=[1.0,1.1,1.0,1.8,1.8]
        if t.cell(0,0).text=='Method' and ncols==5:widths=[1.8,.5,1.8,1.8,.8]
        for col,width in zip(t.columns,widths):col.width=Inches(width)
        pr=t._tbl.tblPr
        borders=OxmlElement('w:tblBorders')
        for side in ['top','left','bottom','right','insideH','insideV']:
            el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
        pr.append(borders)
        margins=OxmlElement('w:tblCellMar')
        for side,val in [('top','80'),('bottom','80'),('left','85'),('right','85')]:
            el=OxmlElement('w:'+side);el.set(qn('w:w'),val);el.set(qn('w:type'),'dxa');margins.append(el)
        pr.append(margins)
        for ri,row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            if ri==0:
                repeat=OxmlElement('w:tblHeader');row._tr.get_or_add_trPr().append(repeat)
            for ci,cell in enumerate(row.cells):
                cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr=cell._tc.get_or_add_tcPr();shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EDF1' if ri==0 else 'FFFFFF');tcpr.append(shade)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1.03
                    if len(t.rows)<=7:p.paragraph_format.keep_with_next=ri<len(t.rows)-1
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 or ncols==3 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.font.name='Times New Roman';run.font.size=Pt(10);run.font.color.rgb=RGBColor(0,0,0)
                        if ri==0:run.bold=True
    # Page numbers support a long mathematical manuscript without distracting headers.
    fp=sec.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');fp._p.append(fld)
    d.core_properties.title='Adaptive Circular Physics Informed Neural Networks for Multiscale Partial Differential Equations'
    d.core_properties.subject='Mathematical analysis and reproducible PDE experiments'
    d.core_properties.author='';d.core_properties.keywords='AC-PINN, compact radial bases, PDE, adaptive Ritz'
    d.save(final)
    print(final)
    print('Native equation objects:',len(d.element.xpath('//m:oMath')))
    print('Words:',len(text.split()))

if __name__=='__main__':main()
