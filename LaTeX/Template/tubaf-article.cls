\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-article}[2024-04-18 TUBAF class for writing a article]

\DeclareKeys[tubaf-article]{%
    footmark.code=\PassOptionsToPackage{\CurrentOption}{tubaf-marks},
    footmark.usage=load,
    title-black.if=@tubaf@article@title@black,
    title-black.usage=load,
    title-trapezoid.if=@tubaf@article@title@trapezoid,
    title-trapezoid.usage=load,
}


\SetKeys[tubaf-article]{
    footmark=author-title,
    title-black=false,
    title-trapezoid=true,
}

\DeclareUnknownKeyHandler[tubaf-article]{%
    \PassOptionsToClass{\CurrentOption}{scrartcl}
}

\ProcessKeyOptions[tubaf-article]

\LoadClass[parskip=half]{scrartcl}

\if@tubaf@article@title@black
\PassOptionsToPackage{black=true}{tubaf-title}
\else
\PassOptionsToPackage{black=false}{tubaf-title}
\fi

\if@tubaf@article@title@trapezoid
\PassOptionsToPackage{trapezoid=true}{tubaf-title}
\else
\PassOptionsToPackage{trapezoid=false}{tubaf-title}
\fi

\RequirePackage{tubaf-logo}
\RequirePackage{tubaf-title}
\RequirePackage{tubaf-marks}
