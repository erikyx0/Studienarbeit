\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-script}[2024-04-18 TUBAF class for writing a protocol]

\DeclareKeys[tubaf-script]{}

\DeclareUnknownKeyHandler[tubaf-script]{%
    \PassOptionsToClass{\CurrentOption}{tubaf-report}
}

\ProcessKeyOptions[tubaf-script]

\LoadClass[footmark=subject-title]{tubaf-report}

\RequirePackage{tabularray}
\RequirePackage{lastpage}
\RequirePackage{scrbase}

\providecaptionname{english}{\tubaf@script@pagecountname}{Page Count}
\providecaptionname{ngerman}{\tubaf@script@pagecountname}{Seitenzahl}

\providecaptionname{english}{\tubaf@script@subject}{Lecture Script}
\providecaptionname{ngerman}{\tubaf@script@subject}{Vorlesungsskript}

\subject{\tubaf@script@subject}

\newcommand{\tubaf@script@course}{}
\newcommand{\tubaf@script@course@short}{}
\newcommand{\course}[2][]{
    \renewcommand{\tubaf@script@course}{#2}
    \IfArgIsEmpty{#1}{
        \renewcommand{\tubaf@script@course@short}{#2}
    }{
        \renewcommand{\tubaf@script@course@short}{#1}
    }
}
\subtitle{\tubaf@script@course}

\newcommand{\tubaf@script@module}{}
\newcommand{\tubaf@script@module@short}{}
\newcommand{\module}[2][]{
    \renewcommand{\tubaf@script@module}{#2}
    \IfArgIsEmpty{#1}{
        \renewcommand{\tubaf@script@module@short}{#2}
    }{
        \renewcommand{\tubaf@script@module@short}{#1}
    }
}
\title[\tubaf@script@module@short]{\tubaf@script@module}

\newcommand{\tubaf@script@semester}{}
\newcommand{\tubaf@script@semester@short}{}
\newcommand{\semester}[2][]{
    \renewcommand{\tubaf@script@semester}{#2}
    \IfArgIsEmpty{#1}{
        \renewcommand{\tubaf@script@semester@short}{#2}
    }{
        \renewcommand{\tubaf@script@semester@short}{#1}
    }
}
\date{\tubaf@script@semester}
