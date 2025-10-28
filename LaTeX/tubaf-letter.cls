\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-letter}[2024-04-18 TUBAF class for writing a letter]

\RequirePackage{lastpage}

\DeclareUnknownKeyHandler[tubaf-letter]{%
    \PassOptionsToClass{\CurrentOption}{scrlttr2}
}
\ProcessKeyOptions[tubaf-letter]

\RequirePackage{tubaf-logo}

\LoadClass[
    10pt,
    backaddress=plain,
    refline=nodate,
    firstfoot=true,
    parksip=half
]{scrlttr2}

\RequirePackage[left=20mm,right=20mm,top=20mm,bottom=25mm]{geometry}
\RequirePackage{adjustbox}
\RequirePackage{tabularray}
\RequirePackage{lastpage}
\RequirePackage{scrlayer-scrpage}

\newcommand{\tubaf@letter@footcontact}{TU Bergakademie Freiberg · Akademiestraße 6 · 09599 Freiberg · tu-freiberg.de}
\newcommand{\footcontact}[1]{\renewcommand{\tubaf@letter@footcontact}{#1}}

\setplength{firstheadhpos}{18mm}
\setplength{firstheadvpos}{13mm}
\setplength{firstheadwidth}{50mm}
\setplength{toaddrhpos}{22.5mm}
\setplength{toaddrvpos}{50mm}
\setplength{toaddrindent}{0mm}
\setplength{toaddrheight}{40mm}
\setplength{backaddrheight}{6mm}

\setplength{refhpos}{40mm}
\setplength{refvpos}{100mm}
\setplength{refwidth}{0mm}

\setplength{locvpos}{17mm}
\setplength{lochpos}{20mm}
\setplength{locwidth}{70mm}
\setplength{locheight}{83mm}

\setplength{firstfoothpos}{20mm}
\setplength{firstfootvpos}{271mm}
\setplength{firstfootwidth}{170mm}

\renewcommand*{\raggedsignature}{\raggedright}

\setkomavar{firsthead}{\adjustbox{width=50mm, valign=T}{\tubaflogo}}

\providecaptionname{english}{\tubaf@letter@pagename}{Page}
\providecaptionname{ngerman}{\tubaf@letter@pagename}{Seite}
\newlength{\logosheight}
\setlength{\logosheight}{13mm}
\newlength{\logosep}
\setlength{\logosep}{5mm}

\setkomavar{firstfoot}{\fontsize{7pt}{13pt}\selectfont\tubaf@letter@pagename\space\thepage/\pageref{LastPage}\hfill\resizebox{!}{\logosheight}{\logoauditfamiliengerechtehochschule}\hspace{\logosep}\resizebox{!}{\logosheight}{\logofamilieinderhochschule}\hspace{\logosep}\resizebox{!}{\logosheight}{\logoweltoffenehochschule}\hspace{\logosep}\resizebox{!}{\logosheight}{\logoinnovativehochschule}\\
\raggedleft\tubaf@letter@footcontact}
\ifoot*{\tubaf@letter@pagename\space\thepage/\pageref{LastPage}}
\cfoot*{}
\ofoot*{\tubaf@letter@footcontact}
\setkomafont{pagefoot}{\fontsize{7pt}{13pt}\selectfont\normalshape}

\providecaptionname{english}{\tubaf@letter@subjectname}{Subject}
\providecaptionname{ngerman}{\tubaf@letter@subjectname}{Betreff}

\providecaptionname{english}{\tubaf@letter@processorname}{Processor}
\providecaptionname{ngerman}{\tubaf@letter@processorname}{Bearbeiter/in}

\providecaptionname{english}{\tubaf@letter@roomname}{Room}
\providecaptionname{ngerman}{\tubaf@letter@roomname}{Raum}

\providecaptionname{english}{\tubaf@letter@phonename}{Phone}
\providecaptionname{ngerman}{\tubaf@letter@phonename}{Telefon}

\providecaptionname{english}{\tubaf@letter@mailname}{E-Mail}
\providecaptionname{ngerman}{\tubaf@letter@mailname}{E-Mail}

\providecaptionname{english}{\tubaf@letter@datename}{Date}
\providecaptionname{ngerman}{\tubaf@letter@datename}{Datum}

\newcommand{\tubaf@letter@faculty}{}
\newcommand{\faculty}[1]{\renewcommand{\tubaf@letter@faculty}{#1}}

\newcommand{\tubaf@letter@institute}{}
\newcommand{\institute}[1]{\renewcommand{\tubaf@letter@institute}{#1}}

\newcommand{\tubaf@letter@group}{}
\newcommand{\group}[1]{\renewcommand{\tubaf@letter@group}{#1}}

\newcommand{\tubaf@letter@processor}{}
\newcommand{\processor}[1]{\renewcommand{\tubaf@letter@processor}{#1}}

\newcommand{\tubaf@letter@room}{}
\newcommand{\room}[1]{\renewcommand{\tubaf@letter@room}{#1}}

\newcommand{\tubaf@letter@phone}{}
\newcommand{\phone}[1]{\renewcommand{\tubaf@letter@phone}{#1}}

\newcommand{\tubaf@letter@mail}{}
\newcommand{\mail}[1]{\renewcommand{\tubaf@letter@mail}{#1}}

\newcommand{\tubaf@letter@subject}{}
\newcommand{\subject}[1]{\renewcommand{\tubaf@letter@subject}{#1}}

\newcommand{\tubaf@letter@backaddress}{}
\newcommand{\backaddress}[1]{\renewcommand{\tubaf@letter@backaddress}{#1}}

\setkomavar{location}{
    \adjustbox{max size={70mm}{15mm}, valign=b}{\tubaf@logo@logo}
    \vskip 5mm
    \fontsize{7pt}{13pt}\selectfont%
    \ifx\tubaf@letter@faculty\empty\else\noindent\textbf{\tubaf@letter@faculty}\\\fi
    \ifx\tubaf@letter@institute\empty\else\noindent\textbf{\tubaf@letter@institute}\\\fi
    \ifx\tubaf@letter@group\empty\else\noindent\tubaf@letter@group\\\fi
    \\
    \noindent\begin{tblr}{colspec={lX},leftsep=0pt, rowsep=0pt}
                 \ifx\tubaf@letter@processor\empty\else\tubaf@letter@processorname: & \tubaf@letter@processor \\\fi
                 \ifx\tubaf@letter@room\empty\else\tubaf@letter@roomname:           & \tubaf@letter@room      \\\fi
                 \ifx\tubaf@letter@phone\empty\else\tubaf@letter@phonename:         & \tubaf@letter@phone     \\\fi
                 \ifx\tubaf@letter@mail\empty\else\tubaf@letter@mailname:           & \tubaf@letter@mail      \\\fi
                 \ifx\@date\empty\else\tubaf@letter@datename:                       & \@date                  \\\fi
    \end{tblr}
}

\setkomavar{subject}{\tubaf@letter@subjectname: \tubaf@letter@subject}
\setkomavar{backaddress}{\tubaf@letter@backaddress}
\setkomavar{signature}{\@author}

\setkomafont{backaddress}{\color{tubaf-primary}}

