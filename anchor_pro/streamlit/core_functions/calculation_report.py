import os
import re
import numpy as np
import pandas as pd
from pylatex import (Package, LineBreak, NewLine, MiniPage, Tabular, LongTable,
                     Document, Command, FlushRight, FlushLeft, LargeText,
                     MediumText, NewPage, Section, Subsection, Subsubsection,
                     Tabularx, Math, MultiColumn, Alignat, Enumerate, MultiRow)
from pylatex.utils import bold, NoEscape
from pylatex.base_classes import Environment

import anchor_pro.config
import anchor_pro.reports.plots as plots
from anchor_pro.elements.concrete_anchors import ConcreteAnchors, ConcreteAnchorResults
from anchor_pro.streamlit.core_functions.parameters import Parameters

# --- Helper Classes and Functions (Mirrored from report.py) ---

class Flalign(Environment):
    """A class to wrap the LaTeX flalign environment."""
    omit_if_empty = True
    packages = [Package("amsmath")]
    _latex_name = 'flalign'

    def __init__(self, numbering=False, escape=False):
        self.numbering = numbering
        self.escape = escape
        if not numbering:
            self._star_latex_name = True
        super().__init__()

def subheader(container, text):
    container.append(NoEscape(r'\smallskip'))
    container.append(LineBreak())
    container.append(NoEscape(rf'\makebox[0pt][l]{{\textit{{\textbf{{{text}}}}}}}'))
    container.append(NewLine())

def subheader_nobreak(container, text):
    container.append(NoEscape(rf'\makebox[0pt][l]{{\textit{{\textbf{{{text}}}}}}}'))
    container.append(NewLine())

def make_figure(sec, width, file, title=None, pos='t', use_minipage=True):
    if use_minipage:
        with sec.create(MiniPage(width=f'{width:.2f}in', pos=pos, align='top')) as mini:
            mini.append(NoEscape(r'\centering'))
            if title:
                mini.append(NoEscape(rf'\textit{{\textbf{{{title}}}}}'))
                mini.append(NewLine())
                mini.append(NoEscape(r'\smallskip'))
            mini.append(NoEscape(rf'\includegraphics[width={width:.2f}in,valign=t]{{ {file} }}\\'))
    else:
        sec.append(NoEscape(rf'\includegraphics[width={width:.2f}in,valign=t]{{ {file} }}\\'))

def utilization_text_color(cell, value, limit):
    color = 'red' if value > limit else 'Green'
    return NoEscape(fr'\textcolor{{{color}}}{{ {cell} }}')

def make_table(sec, title, header, units, data, alignment=None, col_formats=None,
               utilization_cols=[], utilization_limit=1.0,
               rows_to_highlight=None, add_index=True, width=r'\textwidth', pos='t', use_minipage=True,
               font_size='footnotesize', align='l'):
    
    # Convert input data
    if isinstance(data, pd.DataFrame):
        data_values = data.to_numpy()
    elif isinstance(data, dict):
        data_values = list(zip(*data.values()))
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data_values = data[:, np.newaxis]
        else:
            data_values = data
    elif isinstance(data, list):
        data_values = data
    else:
        raise TypeError("Unsupported data type.")

    num_cols = len(header)
    if alignment is None:
        alignment = 'l' + 'c' * (num_cols - 1)

    if add_index:
        header = ['\#'] + header
        units = [''] + units
        alignment = 'c' + alignment
        col_formats = ['{:.0f}'] + col_formats
        utilization_cols = [idx + 1 for idx in utilization_cols]
        data_values = [(i + 1,) + tuple(row) for i, row in enumerate(data_values)]

    # Header formatting
    header[0] = NoEscape(r'\rowcolor{lightgray} ' + header[0])
    units[0] = NoEscape(r'\rowcolor{lightgray} ' + units[0])

    if use_minipage:
        with sec.create(MiniPage(width=width, pos=pos, align=align)) as mini:
            if title:
                mini.append(NoEscape(rf'\textit{{\textbf{{{title}}}}}'))
                mini.append(NewLine())
                mini.append(NoEscape(r'\smallskip'))
            mini.append(NoEscape(f'\\begin{{{font_size}}}'))
            with mini.create(Tabular(alignment)) as table:
                _populate_table(table, header, units, data_values, col_formats, 
                                utilization_cols, utilization_limit, rows_to_highlight)
            mini.append(NoEscape(f'\\end{{{font_size}}}'))
    else:
        if title:
            sec.append(NoEscape(rf'\textit{{\textbf{{{title}}}}}'))
            sec.append(NewLine())
        sec.append(NoEscape(f'\\begin{{{font_size}}}'))
        with sec.create(LongTable(alignment)) as table:
            _populate_table(table, header, units, data_values, col_formats, 
                            utilization_cols, utilization_limit, rows_to_highlight)
        sec.append(NoEscape(f'\\end{{{font_size}}}'))

def _populate_table(table, header, units, data_values,
                   col_formats, utilization_cols, utilization_limit, rows_to_highlight):
    table.add_hline()
    table.add_row(header)
    table.add_row(units)
    table.add_hline()

    for i, row in enumerate(data_values):
        formatted_row = [fmt.format(val) if val is not None else val for fmt, val in zip(col_formats, row)]
        for idx in utilization_cols:
            if idx < len(row) and row[idx] is not None:
                formatted_row[idx] = utilization_text_color(formatted_row[idx], row[idx], utilization_limit)

        if rows_to_highlight is not None:
            if isinstance(rows_to_highlight, (int, np.int_)):
                highlight = (i == rows_to_highlight)
            else:
                highlight = (i in rows_to_highlight)
            
            if highlight:
                formatted_row[0] = NoEscape(r'\rowcolor{yellow} ' + formatted_row[0])

        table.add_row(formatted_row)
        table.add_hline()

# --- Main Report Class ---

class CalculationReport:
    def __init__(self, project_info: dict, parameters: Parameters, results: ConcreteAnchorResults, anchor_obj: ConcreteAnchors):
        self.project_info = project_info
        self.params = parameters
        self.results = results
        self.anchor_obj = anchor_obj
        
        self.logo_path = os.path.join(anchor_pro.config.base_path, "graphics", "DegLogo.pdf").replace('\\', '/')
        self.doc = self.setup_document()

    def setup_document(self):
        geometry_options = {"margin": "1in", "top": "1in", "bottom": "1in"}
        doc = Document(geometry_options=geometry_options, document_options=['fleqn'])
        
        # Preamble setup
        doc.preamble.append(NoEscape(r'\setlength{\headheight}{0.5in}'))
        packages = ['times', 'helvet', 'mathptmx', 'amsmath', 'adjustbox', 'xcolor', 'pgf', 'graphicx', 'hyperref', 'fancyhdr', 'pdfpages', 'array']
        
        for pkg in packages:
            options = None
            if pkg == 'adjustbox': options = 'export'
            if pkg == 'xcolor': options = ['table', 'dvipsnames']
            if pkg == 'hyperref': options = ['hidelinks', 'bookmarksdepth=2', 'bookmarksnumbered']
            doc.packages.append(Package(pkg, options=options))

        doc.preamble.append(NoEscape(r'\usepackage{sectsty}'))
        doc.preamble.append(NoEscape(r'\allsectionsfont{\sffamily}'))
        doc.preamble.append(NoEscape(r'\fancypagestyle{StyleSectionSheet}{'))
        doc.preamble.append(NoEscape(r'\fancyheadoffset{0in}'))
        doc.preamble.append(NoEscape(r'\fancyfootoffset{0in}'))
        doc.preamble.append(NoEscape(rf'\fancyhead[L]{{\includegraphics[height=.4in]{{{self.logo_path}}}}}'))
        
        header_text = self.project_info.get('project_title', 'Anchor Calculation')
        sub_text = self.project_info.get('package_info1', '')
        
        doc.preamble.append(NoEscape(rf'\fancyhead[R]{{\sffamily {header_text} \\ {sub_text} }}'))
        doc.preamble.append(NoEscape(r'\fancyhead[C]{}'))
        doc.preamble.append(NoEscape(r'\fancyfoot[L,R]{}'))
        doc.preamble.append(NoEscape(r'\fancyfoot[C]{\thepage}'))
        doc.preamble.append(NoEscape(r'\renewcommand{\headrulewidth}{1pt}'))
        doc.preamble.append(NoEscape(r'\setlength{\headsep}{0.5in}'))
        doc.preamble.append(NoEscape(r'\renewcommand{\footrulewidth}{1pt}}'))
        doc.preamble.append(NoEscape(r'\renewcommand{\familydefault}{\sfdefault}'))
        doc.preamble.append(NoEscape(r'\setcounter{tocdepth}{2}'))
        doc.preamble.append(NoEscape(r'\everydisplay{\footnotesize}'))

        return doc

    def cover_page(self):
        doc = self.doc
        doc.append(NoEscape(r'\pdfbookmark[0]{Coverpage}{cover}'))
        
        # Logos and Address
        with doc.create(MiniPage(width="4.5in")) as minipage:
            minipage.append(Command('includegraphics', options='width=3in', arguments=NoEscape(self.logo_path)))
        doc.append(NoEscape(r'\hfill'))
        with doc.create(MiniPage(width="1.95in")) as minipage:
            with minipage.create(FlushRight()) as right_aligned:
                right_aligned.append(bold('Degenkolb Engineers'))
                right_aligned.append(NewLine())
                right_aligned.append(self.project_info.get('address', ''))
                right_aligned.append(NewLine())
                right_aligned.append(self.project_info.get('city', ''))
                right_aligned.append(NewLine())

        # Project Info
        with doc.create(FlushLeft()) as fl:
            fl.append(NoEscape(r'\vspace{1in}'))
            fl.append(LargeText(bold(self.project_info.get('project_title', 'Anchor Design Calculation'))))
            fl.append(NoEscape(r' \hrule \medskip '))
            
            for i in range(1, 5):
                val = self.project_info.get(f'project_info{i}')
                if val:
                    fl.append(val)
                    fl.append(NewLine())
            
            fl.append(NoEscape(r'\vfill'))
            
            if self.project_info.get('job_number'):
                fl.append(NoEscape(rf'Job Number: {self.project_info["job_number"]}'))
                fl.append(NewLine())
                
            fl.append(NoEscape(r'\vfill'))
            fl.append(NoEscape(r'\hrule'))
            fl.append(NoEscape(r'\normalsize'))
            fl.append(NewPage())

    def input_summary(self):
        with self.doc.create(Section("Input Summary")):
            self.doc.append(NoEscape(r'\pagestyle{StyleSectionSheet}'))
            
            with self.doc.create(Tabularx('lX', pos='t')) as table:
                table.add_hline()
                table.add_row([NoEscape(r'\rowcolor{lightgray} Calculation Parameters'), ''])
                table.add_hline()
                table.add_row(['Name', self.params.name])
                table.add_row(['Load Mode', self.params.load_mode])
                table.add_row(['Anchor Type', self.params.selected_anchor_id])
                table.add_hline()
                
                # Concrete Properties
                table.add_row([NoEscape(r'\rowcolor{lightgray} Concrete Properties'), ''])
                table.add_hline()
                table.add_row(['Profile', self.params.profile.value])
                table.add_row(['Condition', 'Cracked' if self.params.cracked_concrete else 'Uncracked'])
                table.add_row([NoEscape(r"$f'_c$"), f"{self.params.fc:.0f} psi"])
                table.add_row(['Thickness', f"{self.params.t_slab:.2f} in"])
                table.add_hline()

            self.doc.append(NoEscape(r'\bigskip'))
            
            # Loads Table
            if self.params.load_mode == "Global":
                with self.doc.create(MiniPage(width=r"3in", pos='t', align='l')) as mini:
                    # USE subheader_nobreak HERE:
                    subheader_nobreak(mini, "Applied Global Loads")
                    with mini.create(Tabular('lr')) as table:

                        table.add_hline()
                        for k, v in self.params.loads.items():
                            unit = 'in-lbs' if k.startswith('M') or k == 'T' else 'lbs'
                            table.add_row([f"${k}$", f"{v:.0f} {unit}"])
                        table.add_hline()
            else:
                self.doc.append("Individual anchor loads specified directly.")

    def results_summary(self):
        with self.doc.create(Section("Results Summary")):
            
            # Unity Check
            ok = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if self.results.ok else r'\textcolor{red}{\textbf{\textsf{NG}}}'
            equality = r'\leq' if self.results.ok else r'>'
            
            with self.doc.create(MiniPage(width=r"4in", pos='t')) as mini:
                subheader_nobreak(mini, "Interaction Check")
                mini.append(NoEscape(rf'''The governing interaction equation (ACI 318-19 R17.8):'''))
                with mini.create(Math(inline=False)) as math:
                    math.append(NoEscape(rf'\text{{Unity}} = {self.results.unity:.2f} {equality} 1.0 \quad \text{{{ok}}}'))

            # Anchor Diagram
            self.doc.append(NoEscape(r'\hfill'))
            
            # Generate diagram using existing plot logic
            try:
                fig, width = plots.anchor_basic(None, self.anchor_obj, self.results)
                
                # FIX: Force image to save in the managed output_path
                # We do not add .png extension here assuming vtk_save adds it
                image_name = f"anchor_diagram_{id(self)}"
                save_path = os.path.join(self.output_path, image_name)
                
                # Save to the specific path
                file_path = plots.vtk_save(fig, filename=save_path)
                
                # Ensure forward slashes for LaTeX compatibility
                file_path = file_path.replace('\\', '/')
                
                make_figure(self.doc, width, file_path, title="Anchor Group Diagram", use_minipage=True)
            except Exception as e:
                self.doc.append(f"Error generating diagram: {str(e)}")

            self.doc.append(NewLine())
            self.doc.append(NoEscape(r'\bigskip'))

            # Anchor Forces Table
            subheader(self.doc, "Individual Anchor Results")
            # For n_theta=1 (Streamlit calculator), we take the 0th index
            forces = self.results.forces[:, :, 0] # (n_anchor, 3)
            xy = self.anchor_obj.geo_props.xy_anchors
            
            header = [NoEscape('$x$'), NoEscape('$y$'), 
                      NoEscape('$T$'), NoEscape('$V_x$'), NoEscape('$V_y$'), 
                      'Ten. Unity', 'Shear Unity', 'Combined']
            units = ['(in)', '(in)', '(lbs)', '(lbs)', '(lbs)', '', '', '']
            
            data = []
            for i in range(len(xy)):
                t_unity = self.results.tension_unity_by_anchor[i, 0]
                s_unity = self.results.shear_unity_by_anchor[i, 0]
                c_unity = self.results.unity_by_anchor[i, 0]
                data.append([
                    xy[i,0], xy[i,1],
                    forces[i, 2], forces[i, 0], forces[i, 1],
                    t_unity, s_unity, c_unity
                ])
                
            formats = ['{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}', '{:.0f}', '{:.2f}', '{:.2f}', '{:.2f}']
            make_table(self.doc, None, header, units, data, col_formats=formats, 
                       utilization_cols=[5, 6, 7], width=r'\textwidth', use_minipage=False)

    def detailed_calculations(self):
        # Mirroring concrete_summary_full logic for detail sections
        
        # 1. Spacing
        with self.doc.create(Section("Detail Checks")):
            self._spacing_section()
            self._limit_states_section()

    def _spacing_section(self):
        with self.doc.create(Subsection("Anchor Spacing and Edge Distance")):
            reqs = self.results.spacing_requirements
            
            with self.doc.create(MiniPage(width=r"3.5in", pos='t')) as mini:
                if reqs.slab_thickness_ok:
                    mini.append(NoEscape(r'\textcolor{Green}{\textbf{Slab thickness is sufficient.}}'))
                else:
                    mini.append(NoEscape(r'\textcolor{red}{\textbf{Slab thickness is insufficient.}}'))
                mini.append(NewLine())
                
                if reqs.edge_and_spacing_ok:
                    mini.append(NoEscape(r'\textcolor{Green}{\textbf{Edge distance and spacing requirements are met.}}'))
                else:
                    mini.append(NoEscape(r'\textcolor{red}{\textbf{Edge distance or spacing requirements are NOT met.}}'))
            
            # Add Spacing Diagram
            try:
                fig, width = plots.anchor_spacing_criteria(None, self.anchor_obj, self.results)
                filename = f"spacing_{id(self)}"
                file_path = plots.plt_save(fig, filename=filename)
                self.doc.append(NoEscape(r'\hfill'))
                make_figure(self.doc, width, file_path, use_minipage=True)
            except Exception:
                pass

    def _limit_states_section(self):
        # Limit States Table
        tension_limits = [
            (self.results.steel_tension_calcs, "Steel Tensile Strength", "Tension"),
            (self.results.tension_breakout_calcs, "Concrete Tension Breakout", "Tension"),
            (self.results.anchor_pullout_calcs, "Anchor Pullout", "Tension"),
            (self.results.side_face_blowout_calcs, "Side Face Blowout", "Tension"),
            (self.results.bond_strength_calcs, "Bond Strength", "Tension")
        ]

        shear_limits = [
            (self.results.steel_shear_calcs, "Steel Shear Strength", "Shear"),
            (self.results.shear_pryout_calcs, "Shear Pryout", "Shear"),
            (self.results.shear_breakout_calcs, "Concrete Shear Breakout", "Shear")
        ]

        # Use the governing group index
        t_idx = self.results.governing_tension_group
        s_idx = self.results.governing_shear_group
        theta = 0 # Single load case

        with self.doc.create(Subsection("Limit States")):
            with self.doc.create(Tabularx('Xcrrrrrr')) as table:
                table.add_hline()
                header = [NoEscape(r'\rowcolor{lightgray} Limit State'), 'Mode', NoEscape('$N_u$'), NoEscape('$N_n$'), 
                          NoEscape(r'$\phi$'), NoEscape(r'$\phi_{seis}$'), NoEscape(r'$\phi N_n$'), 'Unity']
                table.add_row(header)
                table.add_hline()

                # Process Tension
                for calc_list, label, mode in tension_limits:
                    if calc_list and calc_list[t_idx]:
                        c = calc_list[t_idx]
                        cap = c.Nsa if hasattr(c, 'Nsa') else c.Ncb if hasattr(c, 'Ncb') else c.Np if hasattr(c, 'Np') else c.Nsbg if hasattr(c, 'Nsbg') else c.Nag
                        if isinstance(cap, np.ndarray): cap = cap[theta]
                        
                        dem = c.demand[theta] if isinstance(c.demand, np.ndarray) else c.demand
                        res_cap = c.phi * c.phi_seismic * cap
                        unity = dem / res_cap if res_cap > 0 else 0
                        
                        row = [label, mode, f'{dem:.0f}', f'{cap:.0f}', f'{c.phi:.2f}', f'{c.phi_seismic:.2f}', f'{res_cap:.0f}', utilization_text_color(f'{unity:.2f}', unity, 1.0)]
                        table.add_row(row)
                        table.add_hline()

                # Process Shear
                table.add_row([NoEscape(r'\rowcolor{lightgray} Limit State'), 'Mode', NoEscape('$V_u$'), NoEscape('$V_n$'), 
                          NoEscape(r'$\phi$'), NoEscape(r'$\phi_{seis}$'), NoEscape(r'$\phi V_n$'), 'Unity'])
                table.add_hline()

                for calc_list, label, mode in shear_limits:
                    if calc_list:
                        # Use s_idx for breakout, t_idx for steel/pryout usually, 
                        # but ConcreteAnchorResults structure implies list per group type
                        # For simplicity in this generic report, we check existence and index
                        idx = s_idx if "Breakout" in label else t_idx # Steel/Pryout usually calc'd on tension groups or all groups
                        
                        if idx < len(calc_list) and calc_list[idx]:
                            c = calc_list[idx]
                            cap = c.Vsa if hasattr(c, 'Vsa') else c.Vcp if hasattr(c, 'Vcp') else c.Vcb
                            if isinstance(cap, np.ndarray): cap = cap[theta]
                            
                            dem = c.demand[theta] if isinstance(c.demand, np.ndarray) else c.demand
                            res_cap = c.phi * c.phi_seismic * cap
                            unity = dem / res_cap if res_cap > 0 else 0
                            
                            row = [label, mode, f'{dem:.0f}', f'{cap:.0f}', f'{c.phi:.2f}', f'{c.phi_seismic:.2f}', f'{res_cap:.0f}', utilization_text_color(f'{unity:.2f}', unity, 1.0)]
                            table.add_row(row)
                            table.add_hline()

    def generate_pdf(self):
        file_name = os.path.join(self.output_path, self.file_name)
        self.doc.generate_pdf(file_name, clean_tex=False)

    def generate_report(self):
        self.cover_page()
        self.input_summary()
        self.results_summary()
        self.detailed_calculations()
        self.generate_pdf()