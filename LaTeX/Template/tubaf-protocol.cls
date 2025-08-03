\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-protocol}[2024-04-18 TUBAF class for writing a protocol]

\DeclareKeys[tubaf-protocol]{}

\DeclareUnknownKeyHandler[tubaf-protocol]{%
    \PassOptionsToClass{\CurrentOption}{tubaf-article}
}

\ProcessKeyOptions[tubaf-protocol]

\LoadClass[titlepage=true]{tubaf-article}

\RequirePackage{tabularray}
\RequirePackage{lastpage}

\providecaptionname{english}{\tubaf@protocol@subject}{Protocol}
\providecaptionname{ngerman}{\tubaf@protocol@subject}{Protokoll}
\subject{\tubaf@protocol@subject}

\providecaptionname{english}{\tubaf@protocol@supervisorname}{Supervisor}
\providecaptionname{ngerman}{\tubaf@protocol@supervisorname}{Betreuer}

\providecaptionname{english}{\tubaf@protocol@coursename}{Course}
\providecaptionname{ngerman}{\tubaf@protocol@coursename}{Studiengang}

\providecaptionname{english}{\tubaf@protocol@datename}{Date}
\providecaptionname{ngerman}{\tubaf@protocol@datename}{Datum}

\providecaptionname{english}{\tubaf@protocol@pagecountname}{Page Count}
\providecaptionname{ngerman}{\tubaf@protocol@pagecountname}{Seitenzahl}

\providecaptionname{english}{\tubaf@protocol@authorsname}{Authors}
\providecaptionname{ngerman}{\tubaf@protocol@authorsname}{Autoren}

\newcommand{\tubaf@protocol@supervisor}{}
\newcommand{\supervisor}[1]{\renewcommand{\tubaf@protocol@supervisor}{#1}}

\newcommand{\tubaf@protocol@course}{}
\newcommand{\course}[1]{\renewcommand{\tubaf@protocol@course}{#1}}

\newcommand{\module}[1]{\subtitle{#1}}
\newcommand{\experiment}[1]{\title{#1}}

\titlefoot{
    \renewcommand{\and}{\\}
    \begin{tblr}{ll}
        \tubaf@protocol@authorsname:    & \tubaf@protocol@author     \\
        \tubaf@protocol@coursename:     & \tubaf@protocol@course     \\
        \tubaf@protocol@supervisorname: & \tubaf@protocol@supervisor \\
        \tubaf@protocol@datename:       & \tubaf@protocol@date       \\
        \tubaf@protocol@pagecountname:  & \pageref{LastPage}         \\
    \end{tblr}
}

\renewcommand{\tubaf@title@titlepage@prehook}{
    \let\tubaf@protocol@author\@author
    \let\tubaf@protocol@date\@date
    \let\@author\empty
    \let\@date\empty

    \renewcommand{\tubaf@title@titlepage@posthook}{
        \let\@author\tubaf@protocol@author
        \let\@date\tubaf@protocol@date
    }
}