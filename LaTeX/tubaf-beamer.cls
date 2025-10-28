\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-beamer}[2024-04-18 TUBAF beamer based presentation class]

\DeclareKeys[tubaf-beamer]{%
    title-dark.if = @tubaf@beamer@title@dark,
    title-dark.usage = preamble,
    title-graphic-opacity.store = \tubaf@beamer@title@graphic@opacity,
    title-graphic-opacity.usage = preamble,
    foot-trapezoid.if = @tubaf@beamer@foot@trapezoid,
    foot-trapezoid.usage = preamble,
    foot-trapezoid-opacity.store = \tubaf@beamer@foot@trapezoid@opacity,
    foot-trapezoid-opacity.usage = preamble,
    color-primary.store = \tubaf@beamer@color@primary,
    color-primary.usage = load,
    color-secondary.store = \tubaf@beamer@color@secondary,
    color-secondary.usage = load,
    color-tertiary.store = \tubaf@beamer@color@tertiary,
    color-tertiary.usage = load,
}

\SetKeys[tubaf-beamer]{
    title-dark=false,
    title-graphic-opacity=0.6,
    foot-trapezoid=false,
    foot-trapezoid-opacity=0.15,
    color-primary=tubaf-primary,
    color-secondary=tubaf-accent-11,
    color-tertiary=tubaf-accent-24,
}

\DeclareUnknownKeyHandler[tubaf-beamer]{%
    \PassOptionsToClass{\CurrentOption}{beamer}
}

\ProcessKeyOptions[tubaf-beamer]

\LoadClass[aspectratio=169]{beamer}

\RequirePackage{scrbase}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\usetikzlibrary{calc}
\RequirePackage[normalem]{ulem}
\RequirePackage{scrbase}
\RequirePackage{tubaf-logo}
\RequirePackage{tubaf-trapezoid}

\setbeamercolor*{palette primary}{fg=\tubaf@beamer@color@primary}
\setbeamercolor*{palette secondary}{fg=\tubaf@beamer@color@secondary}
\setbeamercolor*{palette tertiary}{fg=\tubaf@beamer@color@tertiary}

\setbeamercolor*{background}{parent=palette primary}
\setbeamercolor*{footline}{parent=palette primary}
\setbeamercolor*{structure}{parent=palette primary}
\setbeamercolor*{alerted text}{parent=palette secondary}
\setbeamercolor*{separation line}{parent=palette primary}

\setbeamercolor*{block title}{fg=white,bg=palette primary.fg,use=palette primary}
\setbeamercolor*{block body}{bg=block title.bg!15!white,use=block title}
\setbeamercolor*{block title alerted}{fg=white,bg=palette secondary.fg,use=palette secondary}
\setbeamercolor*{block body alerted}{bg=block title alerted.bg!15!white,use=block title alerted}
\setbeamercolor*{block title example}{fg=white,bg=palette tertiary.fg,use=palette tertiary}
\setbeamercolor*{block body example}{bg=block title example.bg!15!white,use=block title example}

\newcommand{\speaker}[1]{\underline{#1}}
\renewcommand{\and}{\unskip, }
\Ifnotundefined{beamer@andtitle}{\renewcommand{\beamer@andtitle}{\unskip, }}{}

\newcommand{\makeclosing}{
    \usebeamertemplate{closing page}
}

\providecaptionname{english}{\tubaf@closing}{Thank you for your attention!}
\providecaptionname{ngerman}{\tubaf@closing}{Vielen Dank für Ihre Aufmerksamkeit!}

\newcommand{\closing}[1]{\renewcommand{\tubaf@closing}{#1}}

\providecaptionname{english}{\tubaf@contact}{%
    Technical University Bergakademie Freiberg\\
    Akademiestraße 6\\
    09599 Freiberg
}
\providecaptionname{ngerman}{\tubaf@contact}{%
    Technische Universität Bergakademie Freiberg\\
    Akademiestraße 6\\
    09599 Freiberg
}

\newcommand{\contact}[1]{\renewcommand{\tubaf@contact}{#1}}

\newcommand{\footreference}[1]{%
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor{footreference}
        \node[anchor=south, align=center, inner sep=0.1cm, text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}, color=fg] at ($(current page.south) + (0,\footsize)$) {\usebeamerfont{footreference}#1};
    \end{tikzpicture}
}

\setlength{\logoheadsize}{1.5cm}
\setlength{\logoheadmargin}{0.5cm}

\if@tubaf@beamer@title@dark
\setbeamercolor{title background}{bg=black}
\setbeamercolor{title head}{fg=white}
\setbeamertemplate{title logo primary}{\resizebox{!}{\logoheadsize}{\tubaflogowhite}}
\setbeamertemplate{title logo secondary}{
    \typeout\tubaf@logo@logo
    \Ifstr{\tubaf@logo@logowhite}{}{}{\resizebox{!}{\logoheadsize}{\tubaf@logo@logowhite}}
}
\else
\setbeamercolor{title background}{bg=white}
\setbeamercolor{title head}{parent=titlelike}
\setbeamertemplate{title logo primary}{\resizebox{!}{\logoheadsize}{\tubaflogo}}
\setbeamertemplate{title logo secondary}{
    \typeout\tubaf@logo@logo
    \Ifstr{\tubaf@logo@logo}{}{}{\resizebox{!}{\logoheadsize}{\tubaf@logo@logo}}
}
\fi

\ProcessOptionsBeamer

\newlength{\footsize}
\setlength{\footsize}{1cm}

\newlength{\titlefootsize}
\setlength{\titlefootsize}{2.5cm}

\newlength{\closingfootsize}
\setlength{\closingfootsize}{4cm}

\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{section in toc}{\usebeamercolor[fg]{local structure}\inserttocsectionnumber. \usebeamercolor[fg]{normal text}\inserttocsection}


\setbeamersize{description width=0mm}
\setlength{\leftmargini}{0.5cm}
\setlength{\leftmarginii}{0.5cm}
\setlength{\leftmarginiii}{0.5cm}

\setbeamerfont{itemize/enumerate body}{size=\normalsize}
\setbeamerfont{itemize/enumerate subbody}{size=\normalsize}
\setbeamerfont{itemize/enumerate subsubbody}{size=\normalsize}
\setbeamerfont{footline}{size=\tiny}
\setbeamertemplate{page number in head/foot}[framenumber]
\setbeamerfont{page number in head/foot}{size=\normalsize}
\setbeamerfont{closing}{parent=title}
\setbeamerfont{contact}{parent=author, size=\small}

\setbeamertemplate{itemize item}{\tikz{\usebeamercolor{local structure}\useasboundingbox (-0.05,-0.12) rectangle(0.05,0.12);\fill[color=fg] (0,0) circle(0.05)}}
\setbeamertemplate{itemize subitem}{\tikz{\usebeamercolor{local structure}\useasboundingbox (-0.05,-0.12) rectangle(0.05,0.12);\fill[color=fg] (0,0) circle(0.05)}}
\setbeamertemplate{itemize subsubitem}{\tikz{\usebeamercolor{local structure}\useasboundingbox (-0.05,-0.12) rectangle(0.05,0.12);\fill[color=fg] (0,0) circle(0.05)}}

\setbeamertemplate{bibliography item}{}
\setbeamercolor*{bibliography entry author}{}
\setbeamercolor*{bibliography entry title}{}
\setbeamercolor*{bibliography entry location}{}
\setbeamercolor*{bibliography entry note}{}

\setbeamertemplate{sidebar right}{}

\setbeamertemplate{title background}{
    \ifx\inserttitlegraphic\empty\else
    \node[anchor=center, text opacity=\tubaf@beamer@title@graphic@opacity, fill=bg] at (current page.center) {\resizebox{1.01\paperwidth}{!}{\inserttitlegraphic}};
    \fi
}

\setbeamertemplate{title head}{
    \node[anchor=north west, align=left, color=fg, name=title, inner xsep=\Gm@lmargin, text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}] at (logo primary.south west) {\usebeamertemplate{title head text}};
}

\setbeamertemplate{title head text}{%
    \begin{minipage}{\linewidth}
        \raggedright
        \usebeamerfont{title}\inserttitle
    \end{minipage}
    \ifx\insertsubtitle\empty\else
    \\[0.5em]
    \begin{minipage}{\linewidth}
        \raggedright
        \usebeamerfont{subtitle}\insertsubtitle
    \end{minipage}
    \fi
}
\setbeamercolor{title head text}{parent=title head}

\setbeamertemplate{title foot}{
    \node[anchor=south west, inner ysep=0.5cm, inner xsep=\Gm@lmargin, align=left, text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}, color=fg] at (current page.south west) {\usebeamertemplate{title foot text}};
}
\setbeamercolor{title foot}{use=titlelike,fg=titlelike.bg,bg=titlelike.fg}

\setbeamertemplate{title foot text}{%
        {\usebeamerfont{author}\insertauthor}
    \ifx\insertinstitute\empty\else
    \\[0.5em]
    {\usebeamerfont{institute}\insertinstitute}
    \fi
    \ifx\insertdate\empty\else
    \\[0.5em]
    {\usebeamerfont{date}\insertdate}
    \fi
}
\setbeamercolor{title foot text}{parent=title foot}

\setbeamerfont{title}{size=\LARGE}
\setbeamerfont{institute}{size=\normalsize}

\setbeamertemplate{title page}{
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor*{titlelike}
        \useasboundingbox[fill=bg] (current page.south west) rectangle(current page.north east);
        \usebeamertemplate*{title background}
        \tubaftrapezoidbottom[fill=fg]{\titlefootsize}

        \node[anchor=north west, inner sep=\logoheadmargin, name=logo primary] at (current page.north west) {\usebeamertemplate{title logo primary}};
        \node[anchor=north east, inner sep=\logoheadmargin, name=logo secondary] at (current page.north east) {\usebeamertemplate{title logo secondary}};

        \usebeamertemplate*{title head}
        \usebeamertemplate*{title foot}
    \end{tikzpicture}
}

\setbeamerfont{section title}{size=\Large}

\setbeamertemplate{section page}{
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor{section title}
        \node[anchor=south west, font=\usebeamerfont{section title}, color=fg] at ($(current page.west) + (\Gm@lmargin, 0)$) {\insertsection};

        \usebeamercolor{separation line}
        \draw[color=fg] ($(current page.west) + (\Gm@lmargin, 0)$) -- ($(current page.east) - (\Gm@rmargin, 0)$);
    \end{tikzpicture}
}

\setbeamerfont{subsection title}{size=\normalsize}
\setbeamercolor{subsection title}{fg=black}

\setbeamertemplate{subsection page}{
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor{section title}
        \node[anchor=south west, font=\usebeamerfont{section title}, color=fg] at ($(current page.west) + (\Gm@lmargin, 0)$) {\insertsection};

        \usebeamercolor{subsection title}
        \node[anchor=north west, font=\usebeamerfont{subsection title}, color=fg] at ($(current page.west) + (\Gm@lmargin, 0)$) {\insertsubsection};

        \usebeamercolor{separation line}
        \draw[color=fg] ($(current page.west) + (\Gm@lmargin, 0)$) -- ($(current page.east) - (\Gm@rmargin, 0)$);
    \end{tikzpicture}
}

\setbeamertemplate{foottext}{\insertshortauthor \\[1em] \insertshorttitle}

\setbeamerfont{footreference}{size=\tiny}
\setbeamercolor{footreference}{fg=darkgray}

\setbeamertemplate{footline}{
    \let\tubaf@beamer@oldspeaker\speaker
    \let\speaker\relax
    \vspace{\footsize}
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor{footline}
        \usebeamerfont{footline}
        \useasboundingbox (current page.south west) rectangle ($(current page.south east) + (0,\footsize)$);
        \node[anchor=north east, inner sep=0.2cm, font=\usebeamerfont{page number in head/foot}] at ($(current page.south west) + (\Gm@lmargin,\footsize)$) {\usebeamertemplate{page number in head/foot}};
        \draw[color=fg] (\Gm@lmargin, 0) -- (\Gm@lmargin,\footsize);
        \node[anchor=north west, align=left, inner sep=0.2cm] at ($(current page.south west) + (\Gm@lmargin,\footsize)$) {\usebeamertemplate{foottext}};

        \node[anchor=south east, inner sep=0.2cm, name=footseal] at (current page.south east) {\resizebox{!}{\dimexpr\footsize-0.2cm\relax}{\tubafsignet}};
        \Ifstr{\tubaf@logo@logo}{}{}{\node[anchor=south east, inner ysep=0.2cm, name=footlogo] at (footseal.south west) {\resizebox{!}{\dimexpr\footsize-0.2cm\relax}{\tubaf@logo@logo}};}
    \end{tikzpicture}
    \let\spreaker\tubaf@beamer@oldspeaker
}

\setbeamertemplate{background}{
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor{background}
        \if@tubaf@beamer@foot@trapezoid
        \tubaftrapezoidbottom[fill=fg,opacity=\tubaf@beamer@foot@trapezoid@opacity]{\dimexpr\footsize+2mm\relax};
        \fi
    \end{tikzpicture}
}

\setbeamertemplate{closing head text}{%
        {\usebeamerfont{closing}\tubaf@closing}
}
\setbeamercolor{closing head text}{parent=closing head}

\setbeamertemplate{closing foot text}{%
        {\usebeamerfont{contact}\tubaf@contact}
}
\setbeamercolor{closing foot text}{parent=closing foot}

\setbeamertemplate{closing head}{
    \node[anchor=north west, align=left, color=fg, name=closing, inner xsep=\Gm@lmargin, text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}] at (logo primary.south west) {\usebeamertemplate{closing head text}};
}
\setbeamercolor{closing head}{parent=title head}

\setbeamertemplate{closing foot}{
    \node[anchor=south west, inner ysep=0.5cm, inner xsep=\Gm@lmargin, align=left, text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}, color=fg] at (current page.south west) {\usebeamertemplate{closing foot text}};
}
\setbeamercolor{closing foot}{parent=title foot}

\setbeamertemplate{closing page}{
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor*{titlelike}
        \useasboundingbox[fill=bg] (current page.south west) rectangle(current page.north east);
        \usebeamertemplate*{title background}
        \tubaftrapezoidbottom[fill=fg]{\closingfootsize}

        \node[anchor=north west, inner sep=\logoheadmargin, name=logo primary] at (current page.north west) {\usebeamertemplate{title logo primary}};
        \node[anchor=north east, inner sep=\logoheadmargin, name=logo secondary] at (current page.north east) {\usebeamertemplate{title logo secondary}};

        \usebeamertemplate*{closing head}
        \usebeamertemplate*{closing foot}
    \end{tikzpicture}
}
